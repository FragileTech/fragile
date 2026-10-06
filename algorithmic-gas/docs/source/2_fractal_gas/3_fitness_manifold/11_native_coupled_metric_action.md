# Coupled Einstein–Hilbert metric response and the literal rank branch

## 1. The consumed configuration

:::{prf:definition} Coupled native Einstein–Hilbert register
:label: def-ncma-register

Retain the complete registers of {prf:ref}`def-native-complete-execution-record`,
{prf:ref}`def-ng-complete-geometric-record` and
{prf:ref}`def-eh-native-interpretation`. The algorithm in this chapter is the
existing `GasConfig::einstein_hilbert(T,h)`, with the corresponding
`GeometryReward` and zero potential. Its reference values are

$$
 a=3,\quad d=a-1=2,\quad N=500,\quad T=0.33,\quad h=0.002,
 \quad\gamma=1,\quad\nu=3,\quad\beta_{\rm curl}=1.
$$

Here $a$ is the kinetic dimension and $d$ is the dimension after the
actual `DropLast` projection. The reference begins with $x_i=v_i=0$.
The original parameterized preset allows $T>0,h>0$; its other numerical
configuration fields remain explicit arguments when a derivative of one of
those existing fields is considered. A parameter derivative does not replace
the configured metric, curvature, graph force, thermostat or cloning law.

The default metric is `NeighborCovariance`, with relative-trace ridge

$$
 \rho=10^{-5},\qquad b_-=10^{-6},\qquad b_+=\infty.
$$

It retains its pseudoinverse cutoff $d\epsilon_{\mathcal E}\lambda_{\max}$.
The site-rank test separately uses the **f64** tolerance

$$
 \epsilon_{\rm rank}=\max(n_{\rm site},d)\epsilon_{64},\qquad
 \sigma_k>\epsilon_{\rm rank}\sigma_1.                 \tag{NCMA.1}
$$

The default degeneracy policy projects a rank-deficient site cloud into its
principal affine subspace before triangulating it. The geometry frame used
by the covariance estimator still contains the original $d$-dimensional
projected positions. Consequently a rank-one tessellation path does not
change the covariance determinant from a $d\times d$ determinant to a
one-dimensional determinant. Exact duplicate sites use `LiftCliques`.
The domain and boundary are the preset's unbounded ones.

The volume is `SqrtDetMetric`, with $f_V=10^{-12}$. The curvature is the
named `ConformalLaplacian`, with $f_R=10^{-12}$, allocated by the existing
`EinsteinHilbertDensity` reward of scale $\lambda_R$, whose default is one.
For the actual undirected CSR graph write

$$
\begin{split}
 C_i&=k_i^{-1}\sum_{j\sim i}z_{ij}z_{ij}^{T},\\
 \tau_i&=\begin{cases}\operatorname{tr}C_i/d,&\operatorname{tr}C_i>0,\\
                         1,&\operatorname{tr}C_i=0,\end{cases}\\
 u_i&={1\over2d}\log\max(\det g_i,f_R),\\
 v_i&=\sqrt{\max(\det g_i,f_V)},\\
 k_{ij}^{R}&={1\over\sqrt{\max(D_{ij},10^{-8})}+10^{-8}},\\
 D_{ij}&=z_{ij}^{T}(g_i+g_j)z_{ij}/2,\\
 S_i&=\sum_{j\sim i}k_{ij}^{R},\\
 w_{ij}^{R}&=k_{ij}^{R}/\max(S_i,10^{-12}),\\
 R_i&=-2(d-1)\sum_{j\sim i}w_{ij}^{R}(u_j-u_i),\\
 r_i&=\lambda_R R_i v_i,\\
 H&=\sum_i r_i.                                      \tag{NCMA.2}
\end{split}
$$

Here $n_{\rm site}$ is the number of distinct sites and equals $N$ almost
surely at a nondegenerate Gaussian terminal refresh. The $\tau_i=1$
fallback also applies to a nonisolated zero covariance. An isolated row
has $C_i=0,\tau_i=1$ and $R_i=0$.
There is no Voronoi coordinate-cell factor in $v_i$: the source code calls
this quantity the Riemannian density of a unit coordinate cell. The action
$H$ is the sum of the already allocated rewards; no alternative action or
face weight is introduced.

The schedule is the actual `EveryStage`: geometry is refreshed before
fitness, after simultaneous cloning and after kinetics. The post-cloning
graph and its `RiemannianKernelVolume` weights are cached through both Boris
B stages. Each B stage separately evaluates its viscous force, its current
curl fit and its Cayley rotation. The quarter-kick coefficient is $h\nu/4$.
The O innovation is the original independent Gaussian with variance

$$
 q^2=T(1-e^{-2\gamma h}),\qquad t=h/2,\qquad c=e^{-\gamma h}.
                                                               \tag{NCMA.3}
$$

Neither the recorded diffusion field nor a metric square root replaces this
isotropic thermostat. The preset has no velocity cap and no additional
position noise. Potential, cap, boundary or noise-factor changes belong to
their actual different configured kernels and are not assigned the following
Gaussian-position formulas without evaluating those changes.

Cloning uses the actual two uniform Fisher–Yates mutual matchings, including
the odd self companion. The scheduled gate period is twenty, clone
regularizer $\varepsilon_c=0$ and saturation $s_c=1$. Its actual living gate is
$p_{ij}=[(f_j-f_i)/((f_i+\varepsilon_c)s_c)]_0^1$ on enabled steps.
These existing fields remain arguments; where $\varepsilon_c\ge0,s_c>0$
are retained at other configured values, the bound below retains them.
Fitness retains
sample standard deviation plus $10^{-30}$, logistic amplitude two, zero
floor and reward/diversity powers one. Distance regularization is $10^{-30}$.
The accepted plan is simultaneous, with zero positional jitter and the
original identity elastic collision. The seed, stream, recording, masks,
cache, source fields and arithmetic data remain in the complete register.

The results below use the real-coordinate interpretation with the actual
positive comparison constants retained, ideal independent Gaussian/uniform
innovations and enough resources for the indicated successful geometry.
They are not derivatives of floating-point rounding or claims of infinite
moments for a finite pseudorandom sample space. An execution failure retains
its actual error status and is not renamed boundary death or conditioned away.
:::

The distinction between (NCMA.1) and exact affine rank is essential here.
The default finite comparison constant gives an open rank-one branch, rather
than a probability-zero set under a continuous Gaussian law.

## 2. The coupled terminal position law

:::{prf:theorem} Actual position-kernel first variation, including zero-mass plans
:label: thm-ncma-coupled-position-response

Fix an entering all-alive finite state and one scheduled step of
{prf:ref}`def-ncma-register`. Average over the actual joint matching law; do
not replace it by independent donor rows. Conditional on a matching context
$D$, let $p_i(\theta,D)$ be the executed clipped acceptance probability.
For every accepted pattern $C\in\{0,1\}^N$, apply the actual simultaneous
copy/collision map and post-cloning geometry, and let $U_C(\theta,D)$ be
the velocity after the complete first B stage. If $\Pi$ is `DropLast`, put

$$
 m_C=\Pi\{X_C+t(1+c)U_C\},\qquad
 s=tq>0,\qquad M=Nd,
$$

and retain the weight

$$
 \pi_C=\prod_i p_i^{C_i}(1-p_i)^{1-C_i}.
$$

The terminal projected positions have the **exact** conditional law

$$
 Y=m_C+sG,\qquad G\sim N(0,I_M).                         \tag{NCMA.4}
$$

This uses the actual complete first B stage and original O noise. The second
B stage remains executed and changes velocities but does not change the
positions in (NCMA.4). The terminal geometry, curvature and reward are
evaluated on these same positions at the final `EveryStage` refresh.

At a parameter value where the computed $p_i,m_C,s$ have derivatives, the
position marginal is differentiable in total variation. One-sided
derivatives give the corresponding one-sided statement at a clipping or
other piecewise branch. For a fixed bounded Borel position observable $\Psi$,

$$
\begin{split}
 \partial_\theta E\Psi(Y)
 =E_D\sum_C\bigg[&\pi'_C E_G\Psi(m_C+sG)\\
 &+\pi_C E_G\left\{\Psi(m_C+sG)
  \left({m'_C\cdot G\over s}
      +{s'\over s}(|G|^2-M)\right)\right\}\bigg].       \tag{NCMA.5}
\end{split}
$$

The plan derivative is

$$
 \pi'_C=\sum_i(2C_i-1)p'_i
        \prod_{j\ne i}p_j^{C_j}(1-p_j)^{1-C_j}.          \tag{NCMA.6}
$$

In particular (NCMA.6) does not divide by $\pi_C$, $p_i$ or $1-p_i$.
It retains a first-order contribution from a plan of zero mass whenever the
actual one-sided probability derivative creates that plan. Disabled gates
have $p_i=p'_i=0$. The derivatives of $U_C$ are precisely the native
graph-force/curl/Cayley derivatives of
{prf:ref}`thm-ng-native-graph-kick-tangent`; the two separately evaluated B
stages are not conflated.

For a parameter-dependent observable $\Psi_\theta$, add
$E_D\sum_C\pi_C E_G(\partial_\theta\Psi_\theta)(Y)$, whenever this direct
term has the local integrability established below. All position-kernel
statements retain the actual clone-induced equality of positions and every
accepted pattern in the preparation; no independence of prepared rows is
required.
:::

:::{prf:proof}
After the first B stage, A1 gives $X_C+tU_C$, O gives $cU_C+qG^{a}$,
and A2 adds $t(cU_C+qG^{a})$. Thus the final positions before B2 equal
$X_C+t(1+c)U_C+tqG^{a}$. The preset has no later position diffusion,
position boundary modification or velocity-cap effect on these positions.
Projection selects independent components of the original $G^{a}$, giving
(NCMA.4). Conditional on the full donor context, the independent acceptance
uniforms yield the finite product $\pi_C$; conditional simultaneous copying
and collision are deterministic and are included in $X_C,U_C$.

At fixed physical $y$, the Gaussian density derivative is its density times

$$
 {m'_C\cdot(y-m_C)\over s^2}
 +{s'\over s}\left({|y-m_C|^2\over s^2}-M\right).
$$

It belongs to $L^1(dy)$. On a small parameter interval the finite means,
their derivatives, the positive variance and its derivative have finite
computed bounds. Differentiating the explicit Gaussian density in $L^1$
follows by the mean-value formula and Gaussian polynomial domination; it
also follows directly by translating and scaling the Gaussian with those
same finite bounds. There are finitely many matchings and patterns.
Product differentiation gives (NCMA.6), including at probabilities zero and
one. Summing the signed density derivatives proves total-variation
differentiability and (NCMA.5).

This argument differentiates the **law at fixed physical positions**. It
therefore includes changing terminal tessellations, ranks and CSR adjacency
in the observable without differentiating a moving triangulation cell by
cell or discarding its boundary flux. All earlier geometry feedback is in
the actually computed $m'_C,p'_i$. The final claim follows by differentiating
the direct observable term under its stated, subsequently proved domination.
:::

:::{prf:corollary} Full native reward-scale feedback without reward moment assumptions
:label: cor-ncma-reward-scale-feedback

For the preset at fixed entering state, vary the existing reward scale
$\lambda_R>0$. The positional and collision maps conditional on a pattern,
the graph/Cayley maps, $m_C$ and $s$ are independent of this scale.
For any bounded observable of terminal positions and velocities, with its
recorded bookkeeping coordinates held out of the observable, its complete
one-step expectation has the derivative

$$
 \partial_{\lambda_R}E\Psi
 =E_D\sum_C\pi'_C E_G\Psi_C(G).                       \tag{NCMA.7}
$$

It holds also after averaging over **any fixed all-alive entering law**, without an
EH-action moment assumption. With $A_N=\sqrt{N-1}$,
$L_N=2/(1+e^{A_N})$, and
$B_N=(2/L_N)^2 A_N/(2\lambda_R s_c)$,

$$
 \sum_C|\pi'_C|\le2NB_N,\qquad
 |\partial_{\lambda_R}E\Psi|\le2NB_N\|\Psi\|_\infty. \tag{NCMA.8}
$$

Closed scheduled gates have zero response. The statement keeps the actual
constant-channel formula and all clipped gates. It does not assert this
uniform derivative at $\lambda_R=0$.
:::

:::{prf:proof}
Write the unscaled reward vector as $b$, its sample standard deviation as
$\sigma_b$, and $\varepsilon=10^{-30}$. For $\lambda_R>0$,

$$
 z_i={\lambda_R(b_i-\bar b)\over\lambda_R\sigma_b+\varepsilon},
 \qquad
 z'_i={\varepsilon(b_i-\bar b)\over
                     (\lambda_R\sigma_b+\varepsilon)^2}.
$$

The sample-variance identity gives $|b_i-\bar b|\le A_N\sigma_b$.
Thus $|z_i|\le A_N$, and maximizing
$\varepsilon\sigma/(\lambda_R\sigma+\varepsilon)^2$ gives
$|z'_i|\le A_N/(4\lambda_R)$. A constant channel has both expressions
zero. The same $A_N$ bound holds for the unchanged diversity channel.
The native logistic map $L(z)=2/(1+e^{-z})$ obeys
$L\ge L_N$, $L\le2$, and $|(\log L)'|\le1$. Consequently
$L_N^2\le f_i\le4$ and
$|\partial_{\lambda_R}\log f_i|\le A_N/(4\lambda_R)$.
Put $D_N=A_N/(4\lambda_R)$ and $R_N=(2/L_N)^2\ge1$.
For the actual unclipped ratio,

$$
 \partial_{\lambda_R}{f_j-f_i\over(f_i+\varepsilon_c)s_c}
 ={f'_j(f_i+\varepsilon_c)-f'_i(f_j+\varepsilon_c)
     \over s_c(f_i+\varepsilon_c)^2}.
$$

The bounds $|f'_i|\le D_Nf_i$ and $f_j/f_i\le R_N$ give magnitude at most
$2R_ND_N/s_c=B_N$ for every $\varepsilon_c\ge0$: writing
$x=\varepsilon_c/f_i$ bounds
$(f_j/f_i+x)/(1+x)^2\le R_N$.
In the preset this is precisely the derivative of $f_j/f_i-1$.
Clipping cannot increase this bound, including one-sided
derivatives at its boundaries. Summing (NCMA.6) over $C$, first keeping
one factor differentiated, gives $2\sum_i|p'_i|\le2NB_N$.

Conditional on a pattern, the reward scale is not a kinetic force and does
not enter the copying, collision, graph or thermostat formulas. This proves
(NCMA.7). The bound is independent of the entering rewards, their size and
their moments. Dominated differentiation therefore proves the claim for any
fixed entering law. Later-time parameter responses additionally retain the
derivative of that entering law; they are not inferred from a one-step result.
:::

## 3. Genuine integrated metric/action variation on the full-rank contribution

:::{prf:theorem} Complete finite-history native reward-scale response
:label: thm-ncma-finite-history-feedback

Fix an all-alive initial phase law independent of $\lambda_R$, and retain the actual
ordered Einstein–Hilbert updates through $n$ steps. Let $\mathcal I_n$ be
the actual set of scheduled open cloning steps. On any compact reward-scale
interval $0<\lambda_-\le\lambda_R\le\lambda_+<\infty$, the complete native
law of the phase history, original Gaussian innovations, matching contexts and
accepted patterns has one-sided total-variation derivatives. The acceptance
uniforms are integrated in their exact conditional Bernoulli law; they are
not added as extra observed coordinates to this total-variation assertion.
Any execution failures
are retained as their distinguished stopped-execution outcomes; no
conditioning on successful histories or boundary-death interpretation is
used. For a fixed bounded measurable history observable $\Psi$,

$$
 \partial_{\lambda_R}^{+}E\Psi
 =\sum_{\ell\in\mathcal I_n}
 E_{\text{all other native marks}}
       \sum_{C_\ell}\pi'_{C_\ell,\ell,+}
                     \Psi\,\prod_{j\ne\ell}\pi_{C_j,j},
 \qquad
 |\partial_{\lambda_R}^{+}E\Psi|
 \le |\mathcal I_n|\,2N B_N(\lambda_-)\|\Psi\|_\infty .
                                                               \tag{NCMA.24}
$$

The displayed sum is the original finite-history pattern/source integral:
all intermediate fitness and gate factors are evaluated on their actual
conditional histories. Formula (NCMA.6) gives each derivative, including
zero-mass patterns. The left derivative has its corresponding literal
one-sided factors. On a two-sided smooth gate branch these coincide.
In particular a raw action moment is not needed for this complete
finite-history feedback theorem at the reference $\lambda_R=1$.
Parameter-dependent stored reward/bookkeeping coordinates additionally have
their direct observable variation; they are not held fixed implicitly.
The bound is finite-population and finite-horizon, and is not asserted
uniform as $N,n\to\infty$.
:::

:::{prf:proof}
Conditional on the complete accepted pattern sequence, all original Gaussian
draws and both matching sequences, the actual physical coordinate trajectory
is independent of the reward scale. Its metric, graph weights, two separate
Boris force/curl evaluations, isotropic thermostat, copying and elastic
collision use the other fixed parameters. The scale enters only the
computed reward-standardization/fitness and hence the conditional plan
probabilities. Their chronological joint density relative to the original
Gaussian source law, original joint matching laws and counting measure on
patterns is the product of these conditional $\pi$ factors. The phases
on which each factor is evaluated are the actual conditional physical
trajectory, so this factorization assumes neither independent marks at
different times nor independent donor rows.

For every such fixed finite history, each positive-scale factor is a
smooth ratio followed by the actual clipping, with a one-sided derivative
at a clipping boundary. Product differentiation gives (NCMA.24).
The single-step bound (NCMA.8) bounds the sum of the absolute derivatives
of each differentiated factor by $2N B_N(\lambda_-)$ independently of
the history. Integrating later conditional pattern factors gives one;
integrating the earlier factors gives the original earlier law. Thus each
of the $|\mathcal I_n|$ differentiated terms has this total-variation bound.
The corresponding local difference quotients have the same domination
by integrating their positive-scale Lipschitz bounds. Dominated convergence
in the finite pattern/source integral proves total-variation convergence
of those quotients and the complete derivative.

If the configured execution reports an error, its original conditional
pattern/source path stops there. Conditional on that path its physical and
geometry computation is still independent of the reward scale, and its
stopped-record outcome is retained in the same measurable integral.
The real logistic lower bound proved above precludes a new zero-fitness
branch at a finite positive scale. This preserves the actual error
interpretation without a success normalization. Differentiating any
explicitly scale-dependent observation adds its direct term whenever
integrable; the theorem's fixed observation is the phase/mark history.
:::

:::{prf:lemma} Primitive inverse regime and finite Gaussian full-rank action moments
:label: lem-ncma-fullrank-action-moments

Suppose the relative-trace metric parameters satisfy

$$
 {\rho\over d+\rho}>d\epsilon_{\mathcal E},\qquad
 b_-<{1\over d+\rho},\qquad
 b_+>{1\over\rho}\ \hbox{if an upper bound is configured}. \tag{NCMA.9}
$$

Then every noncollapsed covariance, including one of deficient matrix rank,
uses $g_i=(C_i+\rho\tau_iI)^{-1}$ without pseudoinverse deletion or metric
clamping. These inequalities hold for the reference $d=2,\rho=10^{-5}$,
$b_-=10^{-6},b_+=\infty$, with both f32 and f64 epsilon.

Let the terminal projected positions be independent Gaussians of common
variance $s^2I_d$ and arbitrary fixed means. Let $\mathcal A$ be the
**actual event that the site rank is $d$** and the open Delaunay graph is
successfully returned, with $N\ge d+1,d\ge2$. For every $1\le p<d$,

$$
 E[\mathbf1_{\mathcal A}|H|^p]<\infty.                 \tag{NCMA.10}
$$

The bound is uniform when $\rho,s$ stay in positive compact intervals and
the finite means stay bounded. It is an estimate for a component of the
original record law; no rank event is imposed on the executed algorithm.
At fixed positions, varying only $\rho$, the same bound holds for
$\mathbf1_{\mathcal A}|\partial_\rho H|^p$, using one-sided derivatives at
the determinant and weight floors.
:::

:::{prf:proof}
Since $C_i\succeq0$ and $\operatorname{tr}C_i=d\tau_i$, the eigenvalues
of $C_i/\tau_i$ belong to $[0,d]$. Those of the regularized matrix divided
by $\tau_i$ lie in $[\rho,d+\rho]$. Conditions (NCMA.9) give the asserted
cutoff and clamp inequalities simultaneously for every covariance.

On $\mathcal A$, every Delaunay vertex has at least $d$ distinct neighbors.
Let $R_{i,d}$ be the Euclidean distance to its $d$-th closest other row,
and $L=\max_{i,j}|Y_i-Y_j|$. Since $k_i\le N-1$,

$$
 {R_{i,d}^2\over dN}\le\tau_i\le {L^2\over d},\qquad
 {(d+\rho)^{-d}\over\tau_i^d}
 \le\det g_i\le {\rho^{-d}\over\tau_i^d}.             \tag{NCMA.11}
$$

The Gaussian density bound is $b_s=(2\pi s^2)^{-d/2}$. Conditional on
$Y_i=x$, independent remaining rows give the elementary union bound

$$
 P(R_{i,d}<r\mid Y_i=x)
 \le \min\{1,K r^{d^2}\},\qquad
 K={N-1\choose d}(b_s\operatorname{vol}B_1)^d.          \tag{NCMA.12}
$$

It is uniform in $x$ and all means. In particular, for $b<d^2$,

$$
 E R_{i,d}^{-b}\le1+{bK\over d^2-b},\qquad
 E[(\log^-R_{i,d})^q]\le {K\Gamma(q+1)\over d^{2q}}
 \quad(q>0).                                          \tag{NCMA.13}
$$

The second inequality integrates $P(\log^-R_{i,d}>u)\le K e^{-d^2u}$.
Positive logarithms of $L$ have every finite moment from the original
Gaussian moments. A fully primitive bound follows from
$L\le2\max_i|m_i|+2s\sum_i|G_i|$ and
$E|G_i|^q=2^{q/2}\Gamma((d+q)/2)/\Gamma(d/2)$.
Thus (NCMA.11), including both literal determinant floors, bounds all moments
of $U=1+\max_i|u_i|$. It also gives

$$
 v_i\le\sqrt{f_V}+\rho^{-d/2}(dN)^{d/2}R_{i,d}^{-d}.
$$

Choose $p<p'<d$. Hölder, with exponents $p'/p$ and $p'/(p'-p)$, combines
the displayed inverse-distance moment at exponent $dp'<d^2$ with the
finite logarithmic moments. Since the native weight rows have mass at most
one,

$$
 |H|\le4(d-1)|\lambda_R|\,U\sum_i v_i,
$$

proving (NCMA.10) and its stated uniformity by the same explicit bounds.

At fixed positions, $\partial_\rho g_i=-\tau_i g_i^2$. Hence

$$
 |\partial_\rho\log\det g_i|\le d/\rho,\quad
 |u'_i|\le1/(2\rho),\quad |v'_i|\le d v_i/(2\rho),
 \quad |D'_{ij}|\le D_{ij}/\rho.
$$

The distance floor contributes derivative zero on its strict floor branch.
Otherwise $|(\log k_{ij}^R)'|\le1/(2\rho)$. Both the normalized and the
row-sum-floor branch then give
$\sum_j|(w_{ij}^R)'|\le1/\rho$. Therefore
$|R'_i|\le 2(d-1)(2U+1)/\rho$ and
$|H'|\le C_{d,\lambda_R,\rho}(U+1)\sum_i v_i$, with, for example,
$C_{d,\lambda_R,\rho}=2|\lambda_R|(d-1)(d+3)/\rho$.
The same Hölder calculation proves its $L^p$ bound. All inequalities retain
one-sided values at floors; no continuous interpolation of the graph was used.
:::

:::{prf:theorem} Coupled weak first variation of the actual full-rank action component
:label: thm-ncma-integrated-action-response

Apply {prf:ref}`thm-ncma-coupled-position-response` to a ridge interval
satisfying (NCMA.9). Define $H_{\mathcal A}=\mathbf1_{\mathcal A}H$ on the
original terminal record. Its conditional one-step expectation is finite,
and its ridge derivative is

$$
\begin{split}
 \partial_\rho E H_{\mathcal A}
 =E_D\sum_C\big[&\pi'_C E_G H_{\mathcal A}(Y)\\
 &+\pi_C E_G\mathbf1_{\mathcal A}(Y)\partial_\rho H(Y)\\
 &+\pi_C E_G H_{\mathcal A}(Y)\,m'_C\cdot G/s\big].    \tag{NCMA.14}
\end{split}
$$

For a $T,h$ or another admissible existing parameter variation, retain
the additional variance score $(s'/s)(|G|^2-M)$ and all its actual direct
derivatives. For the default first transition from the origin,
$Y=sG,m=0$, all gate probabilities are zero and the complete first B stage
has zero velocity, so

$$
 \partial_T E H_{\mathcal A}
 ={1\over2T}E\left[H_{\mathcal A}\sum_i(|G_i|^2-d)\right]. \tag{NCMA.15}
$$

This formula includes moving terminal ranks and tessellations through the
original Gaussian likelihood; it is not a derivative with fixed adjacency.
For the **entire** literal geometry law, (NCMA.14) also holds with any smooth
compact positional cutoff $\chi(Y)$ supported away from collisions in place
of $\mathbf1_{\mathcal A}$. Thus the original local action has a coupled
distributional first variation on every such configuration chart, including
literal rank-one charts. These are local tests of the existing reward, not
new kinetic actions. Removing either restriction to assert an uncut action
expectation is prohibited by the next theorem.
:::

:::{prf:proof}
The actual rank test and tessellation depend on positions and their fixed
arithmetic/comparison tags, not on the metric ridge. At fixed physical $Y$,
$\mathbf1_{\mathcal A}$ has zero ridge derivative. There are finitely many
successful graph patterns; metric dependence within each is smooth except
the retained floors, where one-sided derivatives exist. Lemma
{prf:ref}`lem-ncma-fullrank-action-moments` supplies an $L^p$ envelope for
$H_{\mathcal A}$ and its ridge derivative, locally uniformly in the finite
Gaussian means and scale. Its conjugate exponent bounds the original
Gaussian linear and quadratic scores. Hölder and the mean-value formula
therefore justify differentiated Gaussian integration and the finite-plan
sum, giving (NCMA.14). The proof for a noise-amplitude variation differentiates
the explicit Gaussian density; its compact mean/variance envelopes give the
same domination. At the origin, copying keeps the coincident positions and
zero velocities, the reward and diversity channels are constant, and both
accepted gates and the first B-stage force/curl values vanish. B2 is still
executed on the realized O input and may have nonzero force; it does not move
the positions. The stated $Y=sG$ and
$s'/s=1/(2T)$ follow from the actual thermostat.

For a compact cutoff away from collisions, every nonisolated covariance has
$\tau_i\ge\delta^2/(dN)$, with $\delta>0$ the minimum pair distance on
the cutoff's support. All coordinates are bounded. Uniform finite bounds
for $u_i,v_i,H,H'_\rho$ then hold over every graph, including paths; isolated
rows have their actual finite fallback. The same Gaussian-density argument
applies without a rank restriction. No boundary regularity of the cutoff's
triangulation cells is required because integration is at fixed positions.
:::

## 4. The actual uncut reward moment obstruction

:::{prf:theorem} The default rank-projected EH action has infinite absolute mean
:label: thm-ncma-literal-rank-action-divergence

Retain the unbounded Einstein–Hilbert preset, $d\ge2,N\ge3$, the positive
rank tolerance (NCMA.1) with $0<\epsilon_{\rm rank}<1$, and `rank_projection`
enabled. Suppose (NCMA.9) holds, as it does at the reference values. For
independent terminal $d$-dimensional Gaussian rows with any finite means
and any $s>0$, the original native action satisfies

$$
 \lambda_R>0\ \Longrightarrow\ E H^+=\infty,
 \qquad\lambda_R<0\ \Longrightarrow\ E H^-=\infty,
 \qquad\lambda_R\ne0\ \Longrightarrow\ E|H|=\infty.     \tag{NCMA.16}
$$

If the execution law contains reported geometry errors, these moment
inequalities denote the unnormalized integrals over successful terminal
reward evaluations. The error outcomes remain distinct; no action value or
success-conditioned transition is assigned to them. The divergence below
already occurs within the successful part of the original law.

Every configuration used in this proof has finite noncoincident coordinates,
finite covariance metric, finite reward and a successfully returned rank-one
path graph. Thus (NCMA.16) is not an inference from an execution error.
It applies to the default $N=500,d=2,T=0.33,h=0.002$ **first terminal
refresh**, and to every conditional Gaussian terminal-position preparation
in (NCMA.4), hence to their full nonnegative clone-pattern mixture.

The native constant-channel/sample normalization and original logistic
fitness do not remove this raw reward singularity. For each individual
successful record their standardized values obey $|z_i|\le\sqrt{N-1}$,
so their actual fitness remains finite and positive in real arithmetic.
The theorem establishes an uncut action moment obstruction, not failure of
every bounded native transition observable or of the weak kernel response.
:::

:::{prf:proof}
Place two rows at $y_0=0,y_1=z$ and the other $N-2$ rows in mutually
separated small boxes about $L_j e_1$, where $0<L_2<\cdots<L_{N-1}$.
At $z=0$ and zero transverse coordinates the site cloud has nonzero first
singular value and zero remaining ones. For sufficiently small fixed boxes
in all transverse directions, and sufficiently small $|z|$, the literal
test gives exactly rank one with strict inequalities. This is an open set:
its allowed transverse width is positive because $\epsilon_{\rm rank}>0$.
Choose $z$ in a fixed cone about $e_1$. By taking the boxes smaller if
necessary, the principal one-dimensional projections have the strict order
$y_0,y_1,y_2,\ldots,y_{N-1}$. The actual rank-one `Auto` tessellation is
therefore a path and row zero has exactly the single neighbor row one.
All these strict properties persist for every $0<|z|<r_0$ in a smaller cone.

The metric frame nevertheless uses the original $d$-dimensional
displacement $z$. With $r=|z|$,

$$
 C_0=zz^T,\quad\tau_0=r^2/d,\quad
 \det g_0=A_{d,\rho}r^{-2d},\quad
 A_{d,\rho}=[(1+\rho/d)(\rho/d)^{d-1}]^{-1}.             \tag{NCMA.17}
$$

For small $r$ neither determinant floor is active at this row, so
$v_0=A_{d,\rho}^{1/2}r^{-d}$ and
$u_0=-\log r+(\log A_{d,\rho})/(2d)$. Row one has a neighbor in the fixed
far boxes; every remaining row has a separated path neighbor. Their
$\tau_i$, metrics, $u_i$ and $v_i$ thus lie in fixed positive finite
bounds, independently of $r$. In particular

$$
 D_{01}={1\over2(1+\rho/d)}+O(r^2),
$$

so the original inverse-distance raw weight and its possibly floored
normalization give $w_{01}^R\ge w_*>0$. It follows that

$$
 R_0v_0\ge c_* r^{-d}\log(1/r)-C_*r^{-d}.
$$

Only row one receives $u_0$ in its curvature sum. Its volume is bounded,
and its contribution is $O(\log(1/r))$; all other contributions are
bounded. Shrinking $r_0$ therefore gives
$\sum_iR_iv_i\ge c r^{-d}\log(1/r)>0$.

Allow $y_0$ to vary in a small fixed box, let the remaining rows vary in
their fixed far boxes, and integrate $z=y_1-y_0$ over the chosen cone.
The independent Gaussian joint density has a strictly positive lower bound
on this bounded region for every finite collection of means and $s>0$.
Thus its integral contains

$$
 c\int_0^{r_0}r^{d-1}r^{-d}\log(1/r)\,dr=\infty.
$$

This proves (NCMA.16), with the appropriate sign of $\lambda_R$.
The rank-one path, relative ridge and determinant floors are the original
ones. No artificial nonlocal graph, altered action, compact noise or zero
rank tolerance was used. Every nonzero-radius point has finite real output.
The first-reference-transition claim follows from $Y=sG$ in (NCMA.15).
Conditional preparations and any nonzero mixture weight have the same
strictly positive Gaussian density, which proves their claim. The final
fitness assertion follows from the sample-variance/logistic bounds already
proved in {prf:ref}`cor-ncma-reward-scale-feedback`.
:::

:::{prf:proposition} Failure of a second action moment even on the actual full-rank branch
:label: prop-ncma-fullrank-second-moment

In the reference projected dimension $d=2$, for $N=3$ or $N\ge5$ and the inverse
regime (NCMA.9), the same terminal Gaussian law has

$$
 E[\mathbf1_{\mathcal A}H^2]=\infty\qquad(\lambda_R\ne0). \tag{NCMA.18}
$$

Thus the finite first moments in (NCMA.10) cannot be upgraded to $L^2$ by
discarding only the positive-tolerance rank-one branch.
:::

:::{prf:proof}
For $N\ge5$, put a tagged hull vertex at zero, two nearby vertices at $\varepsilon u$
and $\varepsilon v$, with $u,v$ in small neighborhoods of $e_1,e_2$,
and place every remaining vertex in separated boxes strictly inside the
positive cone generated by $u,v$, at fixed positive distances. The
inequalities defining the tagged Voronoi cell from the two nearby vertices
are $u\cdot x\le\varepsilon|u|^2/2$ and
$v\cdot x\le\varepsilon|v|^2/2$. Each far vertex has displacement
$z=\alpha u+\beta v$, with $\alpha,\beta>0$ bounded away from zero.
Its inequality is implied by these two when
$\varepsilon(\alpha|u|^2+\beta|v|^2)<|z|^2$. Hence the tagged Delaunay
vertex has exactly two neighbors. The two near vertices each have a far
neighbor: their two-node cluster alone cannot separate their cells from the
far sites in these strict cones. Equivalently the exterior edges of the
empty near triangle continue to triangles containing a far site. These
properties have uniform strict margins after reducing the boxes.

The tagged covariance is $\varepsilon^2 C(u,v)$, with $C(u,v)$ uniformly
positive definite. Its $v_0\asymp\varepsilon^{-2}$ and
$u_0=\log(1/\varepsilon)+O(1)$. Every other covariance has a separated
edge, so its $u_i,v_i$ are uniformly bounded. The two tagged inverse
metric-distance weights remain between positive constants. As in the
preceding proof,

$$
 |H|\ge c\varepsilon^{-2}\log(1/\varepsilon)
$$

for small $\varepsilon$. At least two far boxes are chosen about linearly
independent positive-cone vectors, for example $e_1+e_2$ and $2e_1+e_2$,
with any additional far boxes chosen generically. Together with the tagged
vertex these give a uniform positive ratio of the first two singular values.
The literal rank test is therefore full rank, including at the reference f64
tolerance. A single far vertex would not give this uniform ratio and is not used.
The Gaussian density is bounded below on this bounded event. The pair of
two-dimensional near displacements has four relative coordinates; polar
integration over a fixed open shape sector gives

$$
 \int_0^{\varepsilon_0}\varepsilon^3
      \varepsilon^{-4}\log^2(1/\varepsilon)\,d\varepsilon=\infty.
$$

This proves (NCMA.18) for $N\ge5$. For $N=3$, use a neighborhood of
either nonzero full-rank triangle in {prf:ref}`prop-ncma-native-action-signs`
and shrink both relative displacements by $\varepsilon$. The rank ratio
is fixed, the graph and intrinsic lengths are unchanged, and for small
$\varepsilon$ the determinant floors are inactive. The original action
then has magnitude at least $c\varepsilon^{-2}$ on this open shape sector.
Its four relative coordinates give $\int_0\varepsilon^{-1}d\varepsilon=\infty$.
No statement for $N=4$ is needed or inferred from the hull construction.
This proves (NCMA.18) for the original same-record graph and action.
:::

## 5. Literal material scaling and evaluated action signs

:::{prf:theorem} Native material action response, including determinant floors
:label: thm-ncma-material-action-response

On a successful graph in the inverse regime with positive covariance trace
at each nonisolated row, apply the common
coordinate dilation $Y\mapsto e^\theta Y$. It preserves the actual rank
test, tessellation, intrinsic edge lengths and normalized inverse-distance
weights. Away from equality with the determinant floors put
$\chi_i^R=\mathbf1_{\det g_i>f_R}$ and
$\chi_i^V=\mathbf1_{\det g_i>f_V}$. The original material derivative is

$$
\begin{split}
 \dot u_i&=-\chi_i^R,\qquad \dot v_i=-d\chi_i^Vv_i,\\
 \dot H&=\lambda_R\sum_i v_i\left[
 2(d-1)\sum_jw_{ij}^R(\chi_j^R-\chi_i^R)
       -d\chi_i^VR_i\right]                            \tag{NCMA.19}\\
 &=-dH+\lambda_R\left[d\sum_i(1-\chi_i^V)R_iv_i
  +2(d-1)\sum_{i,j}v_iw_{ij}^R(\chi_j^R-\chi_i^R)\right].
\end{split}
$$

On the all-unfloored branch $H(e^\theta Y)=e^{-d\theta}H(Y)$ while that
branch persists. This is the scaling of the literal allocated action; it
is not the Regge $e^{(d-2)\theta}$ scaling of a coordinate-cell curvature
integral. The two quantities use different configured estimators and measures.

For the finite full-rank expectation from the original origin start,

$$
 \partial_T EH_{\mathcal A}={1\over2T}E[\mathbf1_{\mathcal A}\dot H].
                                                               \tag{NCMA.20}
$$

Thus (NCMA.15) and (NCMA.20) are the actual Gaussian-score/material-action
correspondence with every determinant-floor correction retained. Neither
side is assigned an uncut finite value in the regime of (NCMA.16).
:::

:::{prf:proof}
Covariances and traces scale by $e^{2\theta}$, so relative-ridge metrics
scale by $e^{-2\theta}$. The rank comparison scales both singular values
and tolerance by the same factor; a Delaunay dilation preserves its exact
incidence, including a rank-projected path. Intrinsic squared lengths are
therefore unchanged, including their distance and row-sum floor branches.
The determinant scales by $e^{-2d\theta}$, proving the two floor-sensitive
derivatives. Differentiate the literal Laplacian and the reward product to
obtain (NCMA.19). If neither floor is active, all $u_i$ acquire the same
additive constant and the curvature is unchanged, proving the exact scaling.

At the origin $Y=sG$, varying $T$ multiplies positions by
$\sqrt{T/T_0}$. The rank event is invariant under this dilation. The
$L^p$ bounds of {prf:ref}`lem-ncma-fullrank-action-moments` also bound
(NCMA.19) uniformly on a compact temperature interval. Dominated material
differentiation gives (NCMA.20); the preceding Gaussian-density calculation
gives (NCMA.15). This proves the correspondence without an assumed
continuum Einstein equation or a replacement noise factor.
:::

:::{prf:proposition} The original action has both signs on actual full-rank Gaussian records
:label: prop-ncma-native-action-signs

At the default $d=2,\rho=10^{-5}$, metric bounds, determinant floors and
weight constants, the allowed $N=3$ full-rank configurations

$$
 Y_0=(0,0),\quad Y_1=(1,0),\quad Y_2=(0,L)
$$

give, at $\lambda_R=1$,

$$
\begin{array}{c|c}
L& H\\ \hline
1000 &[5.257\,10^{-5},\ 5.259\,10^{-5}]\\
2000 &[-1.581\,10^{-5},\ -1.580\,10^{-5}].
\end{array}                                             \tag{NCMA.21}
$$

The intervals are outward bounds, not a rounding-dependent sign test.
Both configurations have a strict full-rank, inverse, unfloored branch.
Each sign therefore holds on an open event of positive probability under
every nondegenerate terminal Gaussian preparation with $N=3$. This
parameter value is an existing walker-count choice, not an assertion that
the reference $N=500$ swarm is a triangle.
:::

:::{prf:proof}
For the triangle the covariances are exactly

$$
 C_0={1\over2}\begin{pmatrix}1&0\\0&L^2\end{pmatrix},\quad
 C_1=\begin{pmatrix}1&-L/2\\-L/2&L^2/2\end{pmatrix},\quad
 C_2=\begin{pmatrix}1/2&-L/2\\-L/2&L^2\end{pmatrix}.
$$

Use $\tau_i=\operatorname{tr}C_i/2$ and inverse the rational matrices
$C_i+10^{-5}\tau_iI$. All determinant floors are inactive: at $L=2000$
the determinants belong respectively to
$[4.7618,4.7620]10^{-8},[4.7618,4.7620]10^{-8},[1.2345,1.2347]10^{-8}$;
the determinants at $L=1000$ are larger. The actual rank ratios are of
order $1/L$, much larger than $3\epsilon_{64}$.

An exact convenient sign formula, retaining the row-sum floor, is

$$
 H=2(d-1)\lambda_R\sum_{i<j}k_{ij}^R
 \left({v_i\over\max(S_i,10^{-12})}
       -{v_j\over\max(S_j,10^{-12})}\right)(u_i-u_j).    \tag{NCMA.22}
$$

It follows by pairing the two directed terms in (NCMA.2). For these
triangles all row sums exceed one. Rational matrix inversion followed by
outward square-root and logarithm bounds in (NCMA.22) gives (NCMA.21).
For a reproducible elementary interval calculation, bound each square root
by rational bisection to width $10^{-14}$, normalize each positive
logarithm argument to $2^km$, $1\le m\le2$, and use

$$
 \log m=2\sum_{j=0}^{29}{z^{2j+1}\over2j+1}+E_{30},\quad
 z={m-1\over m+1},\quad
 0\le E_{30}\le {2z^{61}\over61(1-z^2)}.
$$

The same series at $z=1/3$ bounds $\log2$; negative $k$ reverses its
interval endpoints. These rational bounds are more than sufficient for
the displayed outward action intervals. Graph incidence and every relevant
threshold have strict margins, so continuity proves the open-event claim.
Gaussian full support proves positive probability. No independent metric
or graph sample is substituted for this original record.
:::

:::{prf:proposition} Evaluated action derivative and the original position likelihood
:label: prop-ncma-original-score-action-distinction

At the origin-start first step of the actual preset, vary only its existing
relative ridge $\rho$ through a positive interval satisfying (NCMA.9).
For every walker count for which the initial geometry is successfully
defined, the complete terminal position law is independent of $\rho$.
Its original position likelihood score is therefore identically zero.
For the allowed $N=3$ triangle with $L=2000$ in (NCMA.21), the actual
allocated action has, at $\rho=10^{-5},\lambda_R=1$,

$$
 \partial_\rho H\in[-1.400,-1.398].                       \tag{NCMA.23}
$$

This nonzero derivative holds on an open terminal-record event of positive
Gaussian probability. Thus the original action's metric derivative is not
the original position log-likelihood score. The genuine integrated
correspondence on the full-rank contribution is instead
$\partial_\rho EH_{\mathcal A}
=E[\mathbf1_{\mathcal A}\partial_\rho H]$ at this first step.
The vanishing score does not assert that a parameter-dependent deterministic
metric payload has a nonsingular full-record density, or that later
geometry-dependent preparations have zero score.

:::

:::{prf:proof}
At the coincident zero-velocity start every reward/diversity channel is
constant, so all accepted-clone probabilities vanish. The actual graph
force and curl of the first B stage vanish for every ridge value. A1 is zero
and the original O/A2 positions are $Y=sG$, with $s=tq$ independent of $\rho$.
B2 is still separately executed and may have nonzero graph force, but it
does not change those positions. This proves the position-likelihood assertion.

For the triangle, set $a_i=v_i/S_i$. Every branch is strict and the
matrices in the preceding proof are rational in $\rho$. Their derivatives
are the literal ones

$$
 g'_i=-\tau_i g_i^2,\quad
 \ell'_i=\operatorname{tr}(g_i^{-1}g'_i),\quad
 u'_i=\ell'_i/4,\quad v'_i=v_i\ell'_i/2,\quad
 a'_i=v'_i/S_i-v_iS'_i/S_i^2,
$$

where $\ell_i=\log\det g_i$ and

$$
 (k^R_{ij})'=
 -{D'_{ij}\over
       2\sqrt{D_{ij}}(\sqrt{D_{ij}}+10^{-8})^2},
 \qquad
 D'_{ij}=z_{ij}^{T}(g'_i+g'_j)z_{ij}/2 .
$$

Differentiate the exact pair identity (NCMA.22):

$$
 H'=2\sum_{i<j}\left[
 (k^R_{ij})'(a_i-a_j)(u_i-u_j)
 +k^R_{ij}(a'_i-a'_j)(u_i-u_j)
 +k^R_{ij}(a_i-a_j)(u'_i-u'_j)\right].
$$

The same rational inverses, square-root bisections and logarithmic remainder
bounds specified in the preceding proof give (NCMA.23); no derivative of
an approximate numerical action is used. For example their intermediate
$u'_0,u'_1,u'_2$ respectively belong to
$[-23809.650,-23809.648]$,
$[-23809.651,-23809.648]$ and
$[-24691.485,-24691.481]$, and the exact final interval is separated from
zero. Strict branch margins give the open event. Equation (NCMA.14) with
the zero mean/gate derivatives proves the final integrated identity.
:::

## 6. Scope of the coupled gravitational identification

The new results establish the actual coupled position-kernel response,
zero-mass clone-plan derivatives, a global bounded-observable reward-scale
response, and an integrated ridge/action response on the full-rank component
and on every compact collision-free configuration test. The original
Gaussian score includes the earlier native graph-force preparation and all
terminal retessellations. The original determinant floors enter the exact
material scaling, and the original action has explicitly evaluated signs.

They also prove that the unbounded preset's uncut EH action has infinite
absolute first moment from its actual positive-tolerance rank-one branch,
and that even its full-rank component has no second moment in projected
dimension two for $N=3$ or $N\ge5$, including the reference $N=500$.
The raw reward remains pointwise finite on those successful
records and its original standardized/logistic fitness remains bounded.
Thus action-integrability failure and existence of a bounded native update
are distinct mathematical statements about the same algorithm.

At $\lambda_R=0$ the allocated action is identically zero. In projected
dimension one the literal curvature factor $d-1$ makes it zero. With two
rows the equal one-edge covariance metrics make the literal curvature zero.
These included regimes remove the moment obstruction by giving a trivial
action; they are not nonzero gravitational endpoints. For $d\ge2,N\ge3$,
positive temperature/timestep, nonzero reward scale and the reference rank
projection/relative metric, (NCMA.16) identifies the failure directly from
the original parameters. Setting rank projection off produces the original
error on its open deficient-rank region; it is not a conservative corrected
kernel. Fixed finite precision and resources retain their actual numeric
errors and cannot be used as an unbounded-space moment proof.

No local Einstein tensor is identified with the graph conformal-Laplacian
reward by these calculations. Such an identification would additionally
have to match its actual measure, covariance metric, curvature estimator,
full evolution and physical reconstruction. An uncut finite expected action
or square-integrable stress in the proved failing preset regime cannot be
supplied by an unproved uniform tail assumption.
