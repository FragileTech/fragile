# The Geometric Gas: Regularity, Stability, and Entropy

(sec-gg-intro)=
## 1. Local measurements and the geometric coefficients

:::{div} feynman-prose
The Geometric Gas changes how walkers move by using the measured landscape to choose forces and noise directions. The mathematical questions have a definite order. First, are those coefficients defined and regular? Second, do they keep the process from escaping? Third, which probability law is stationary, and which functional inequality controls it?

This chapter proves the spectral, derivative, moment, and transport estimates needed for those questions. It then combines them with the complete entropy arguments in {doc}`10_kl_hypocoercive` and {doc}`15_kl_convergence`. The resulting LSI concerns a specified joint law and a full phase-space gradient. Hypocoercive convergence also requires the derivative estimate for the actual evolution, including cloning and survival normalization.

The geometric calculations at the end concern a smooth metric and its connection. They describe holonomy, expansion, and moving cell boundaries. Identifying those objects with a continuum limit uses the hypotheses in {doc}`../3_fitness_manifold/01_emergent_geometry` and the continuum chapters.
:::

(sec-gg-rho-pipeline)=
### 1.1. Normalized local statistics

:::{prf:definition} Localization kernel
:label: def-gg-localization-kernel

Let $\rho>0$ and use a positive symmetric kernel, for example

$$
K_\rho(x,y)=\exp[-|x-y|^2/(2\rho^2)].
$$

Only normalized ratios of these weights enter the finite-swarm statistics. No integral normalization is required. On a domain with boundary, dividing by $\int K_\rho(x,y)dy$ preserves row normalization but generally destroys symmetry; that normalized kernel must be distinguished from $K_\rho$.
:::

:::{prf:definition} Localized moments
:label: def-gg-rho-moments

For a nonempty alive set $A$ of size $k$, a query point $x$, and a measurement $d$ evaluated at $x_j$, define

$$
w_j(x)=\frac{K_\rho(x,x_j)}{\sum_{\ell\in A}K_\rho(x,x_\ell)},\quad
\mu_\rho(x)=\sum_jw_j(x)d(x_j),\quad
s_\rho^2(x)=\sum_jw_j(x)[d(x_j)-\mu_\rho(x)]^2.
$$

For a floor $s_*>0$, set

$$
s'_\rho(x)=\sqrt{s_\rho^2(x)+s_*^2},
\qquad Z_\rho(x)=\frac{d(x)-\mu_\rho(x)}{s'_\rho(x)}.
$$

The weights sum to one, and $s'_\rho\geq s_*$. These functions are as smooth as their measurements and positive kernel denominators permit. Derivatives with respect to a query point holding the swarm fixed differ from derivatives that also move a swarm coordinate; the chosen convention is part of the coefficient definition.
:::

:::{prf:proposition} Limits of finite-swarm Gaussian weights
:label: prop-gg-rho-limits

For a fixed finite configuration, as $\rho\to\infty$, $w_j(x)\to1/k$ and the localized mean and variance converge to their global empirical counterparts. As $\rho\downarrow0$, the weights converge to the uniform distribution on the candidates minimizing $|x-x_j|$. If the query is a unique included particle location, its weight tends to one.
:::

:::{prf:proof}
For the first limit, each exponential tends to one. For the second, divide numerator and denominator by the exponential corresponding to the smallest squared distance. Terms with a positive distance gap vanish, and the minimizing terms remain equal. Substitute the limiting weights in the finite moment sums.
:::

:::{div} feynman-prose
A smaller neighborhood changes the comparison set. At a fixed finite population, the small-scale limit selects the nearest candidates; it is not automatically a continuum delta-kernel limit. Likewise, global averaging removes localization from the weights but can leave spatial derivatives of the measurement itself. It does not automatically switch off the adaptive force or make the diffusion constant.
:::

:::{prf:definition} Local fitness field
:label: def-gg-fitness-potential

A smooth field used in this chapter is

$$
V_i(S)=\eta^{\alpha+\beta}
\exp[\alpha Z_\rho(R,x_i)+\beta Z_\rho(d_{\mathrm{alg}},x_i)],
\qquad \eta>0,\quad\alpha,\beta\geq0.
$$

The measurements, companion averaging, and differentiation convention must be specified before forming
$F_i=\epsilon_F\nabla_{x_i}V_i$ and $H_i=\nabla_{x_i}^2V_i$.
A sampled nonlinear fitness and its expectation are distinct from inserting expected measurements into this exponential. The sampled algorithm is defined in {doc}`../1_the_algorithm/02_fractal_gas_latent`.
:::

:::{prf:lemma} Bounds inherited from normalized measurements
:label: lem-gg-adaptive-force-bounded

Suppose, on the configuration family under consideration, the measurement values and derivatives are bounded, the standard-deviation floor is positive, and

$$
\frac{\sum_j\|D^r a_j\|}{\sum_j a_j}\leq A_r,
\qquad a_j=K_\rho(x,x_j),\qquad r=1,2,3.
$$

Then the quotient, moment, square-root, and exponential chain rules give finite bounds $K_{V,r}$ for $\|D^rV_i\|$, $r\leq3$, and $|F_i|\leq\epsilon_F K_{V,1}$. The constants are independent of $N$ when all displayed inputs are independent of $N$.
:::

:::{prf:proof}
The derivatives of $a_j/\sum_\ell a_\ell$ are controlled by the displayed ratios. Apply the complete quotient estimates in {prf:ref}`lem-normalized-weight-derivatives`, then the moment and Z-score estimates in {prf:ref}`lem-variance-gradient`, {prf:ref}`lem-variance-hessian`, and {prf:ref}`lem-normalized-zscore-bounds`. Since $s'_\rho\geq s_*$, their denominators remain bounded away from zero. The ordinary exponential chain rule completes the estimate for this $V_i$.
:::

:::{prf:remark} Uniformity of the measurement bounds
:label: axiom-gg-bounded-adaptive-force

A bounded-diameter Gaussian family with bounded measurements gives the normalized derivative bounds just stated; see {prf:ref}`cor-normalized-bounded-distance-uniformity`. On unbounded configuration families one must retain the actual weighted ratios. Smoothness alone is not a uniform bound, and a confinement moment estimate does not compactify the support. Bounds for derivatives in the full swarm state must include their particle-index sums or operator norms.
:::

(sec-gg-hybrid-sde)=
## 2. The SDE, spectral bounds, and coefficient calculus

:::{div} feynman-prose
The diffusion coefficient is an inverse square root of a matrix. Its existence is therefore a spectral question. A positive shift repairs negative Hessian eigenvalues only when the shift exceeds their magnitude. Once that margin is positive, matrix calculus gives explicit derivative bounds.

The noise acts in velocity directions. Uniform ellipticity below always refers to that velocity block. Position regularity comes from the kinetic bracket calculation, not from direct position noise.
:::

:::{prf:definition} Geometric kinetic dynamics and cloning
:label: def-gg-sde

Between jumps, on a continuous status stratum,

$$
\begin{aligned}
dx_i&=v_i\,dt,\\
dv_i&=[-\nabla U(x_i)+F_i(S)-\gamma v_i
+\nu\sum_{j\ne i}W_{ij}(X)(v_j-v_i)]dt
+\Sigma_i(S)\circ dW_i,
\end{aligned}
$$

where $W_{ij}=K_{ij}/d_i$, $d_i=\sum_{j\ne i}K_{ij}$, $K_{ij}=K_{ji}>0$, and

$$
g_i=H_i+\epsilon_\Sigma I,\qquad
\Sigma_i=g_i^{-1/2},\qquad D_i=\Sigma_i\Sigma_i^{\mathsf T}=g_i^{-1}.
$$

A specified jump kernel $r_N(S,dS')$ contains the complete cloning update. Killing is represented separately by an interior rate or an absorbing boundary. The continuous jump realization and the actual discrete update are different operators; estimates below name the operator to which they apply.
:::

(sec-gg-axioms)=
### 2.1. Analytic hypotheses

:::{prf:assumption} Confinement of the stable force
:label: axiom-gg-confining-potential

On $\mathbb R^d$, $U\in C^2$ is confining. The explicit quadratic drift calculation below uses $U(0)=0$, $\nabla U(0)=0$, and

$$
m_UI\preceq\nabla^2U\preceq M_UI,\qquad m_U>0.
$$

Nonconvex confining potentials can instead use a proved backbone Lyapunov inequality and the perturbation transfer in {prf:ref}`thm-gg-foster-lyapunov-drift`. Their static Gibbs LSI is supplied by {prf:ref}`thm-nonconvex-main`.
:::

:::{prf:assumption} Friction
:label: axiom-gg-friction

The friction coefficient satisfies $\gamma>0$. Diffusion intensities and friction are kept separate; the covariance of a frozen isotropic Ornstein–Uhlenbeck process is $\sigma^2I/(2\gamma)$.
:::

:::{prf:assumption} Cloning estimate for the chosen functional
:label: axiom-gg-cloning

For a continuous jump realization, use the actual quantity

$$
JW(S)=\int[W(S')-W(S)]r_N(S,dS').
$$

For a discrete cloning update, use $P_{\mathrm{clone}}W-W$. The component estimates in {doc}`03_cloning` can be inserted when the companion, fitness, jitter, status rules, and Lyapunov functional match. A spatial variance contraction alone is not an estimate for every component of $W$.
:::

:::{prf:assumption} Spectral margin
:label: axiom-gg-ueph

For the specified family of configurations,

$$
-\Lambda_- I\preceq H_i(S)\preceq\Lambda_+I,
\qquad a_*:=\epsilon_\Sigma-\Lambda_->0.
$$

Uniformity in $N$ means that $\Lambda_\pm$ and $a_*^{-1}$ are bounded independently of $N$.
:::

:::{prf:assumption} Viscous kernel and degree comparison
:label: axiom-gg-viscous-kernel

The symmetric kernel is sufficiently smooth, each degree is positive, and any use of an unweighted norm records

$$
R_d:=\frac{\max_i d_i}{\min_i d_i}.
$$

A uniform degree-ratio bound holds, for example, when $0<k_*\leq K_{ij}\leq k^*$ for all distinct pairs: $R_d\leq k^*/k_*$. A purely decaying kernel on an unbounded domain need not have such a global bound.
:::

(sec-gg-uniform-ellipticity)=
### 2.2. Velocity ellipticity and matrix derivatives

:::{prf:theorem} Uniform ellipticity from the spectral margin
:label: thm-gg-ueph-construction

Under {prf:ref}`axiom-gg-ueph`,

$$
c_{\min}I\preceq D_i\preceq c_{\max}I,
\qquad c_{\min}=\frac1{\epsilon_\Sigma+\Lambda_+},
\qquad c_{\max}=\frac1{a_*}.
$$

These constants are independent of $N$ precisely when the stated spectral bounds are.
:::

:::{prf:proof}
The eigenvalues of $g_i$ lie in $[a_*,\epsilon_\Sigma+\Lambda_+]$. Invert those positive eigenvalues. The positive square root then defines $\Sigma_i$.
:::

(sec-gg-appendix-a)=
:::{prf:lemma} Derivative of the inverse square root
:label: lem-gg-lipschitz-sigma

If $g\succeq a_*I$ and $\|Dg\|\leq K_3$ in a specified derivative norm, then

$$
\|D(g^{-1/2})\|\leq\frac{K_3}{2a_*^{3/2}},
\qquad \|D(g^{-1})\|\leq\frac{K_3}{a_*^2}.
$$
:::

:::{prf:proof}
Use the matrix resolvent formula

$$
g^{-1/2}=\frac1\pi\int_0^\infty t^{-1/2}(g+tI)^{-1}dt.
$$

Its directional derivative in $E$ is

$$
D(g^{-1/2})[E]=-\frac1\pi\int_0^\infty
 t^{-1/2}(g+tI)^{-1}E(g+tI)^{-1}dt.
$$

The operator norm is at most $\|E\|\pi^{-1}\int_0^\infty t^{-1/2}(a_*+t)^{-2}dt=\|E\|/(2a_*^{3/2})$. For the inverse, $D(g^{-1})[E]=-g^{-1}Eg^{-1}$. No commutation of $g$ and $E$ is required.
:::

:::{prf:lemma} Stratonovich correction for velocity noise
:label: lem-gg-geometric-drift

Let $B_\ell(S)$ be the full phase-space noise vector fields. The Itô correction is

$$
b_{\mathrm{geo}}=\frac12\sum_\ell(DB_\ell)B_\ell.
$$

If the noise has zero position component and its coefficients depend only on positions, then $b_{\mathrm{geo}}=0$. If they also depend on velocities, the derivatives in this formula act in the noisy velocity directions and must be included in the drift.
:::

:::{prf:proof}
The conversion formula differentiates each noise field along itself. For $B_\ell=(0,\Sigma_{:\ell}(X))$, this direction has no position component, whereas $\Sigma$ has no velocity derivative; the directional derivative vanishes. In general it is the displayed contraction, which need not equal $\tfrac12\operatorname{div}_xD$.
:::

:::{prf:definition} Backward generator
:label: def-gg-generator-decomp

With the Itô correction included, the continuous conservative generator is

$$
LF=\sum_i\left[v_i\cdot\nabla_{x_i}F+
(-\nabla U_i+F_i-\gamma v_i-\nu(L_XV)_i+b_{\mathrm{geo},i})\cdot\nabla_{v_i}F
+\frac12D_i:\nabla_{v_i}^2F\right]+JF,
$$

where $L_X=I-W$. Relative to isotropic kinetic diffusion $\sigma^2I$, the diffusion perturbation is $\tfrac12\sum_i(D_i-\sigma^2I):\nabla_{v_i}^2$. Drift estimates apply this backward generator to a test function. The adjoint $L^*$ evolves densities.
:::

:::{prf:corollary} Strong solutions up to and beyond localization
:label: cor-gg-well-posedness

Suppose the Itô coefficients are locally Lipschitz on the state space, the spectral margin holds locally, and jump rates are locally bounded with a specified nonexplosive construction. There is a unique strong solution up to its explosion time. If a coercive $W\geq1$ satisfies $LW\leq CW$ for the full conservative generator, the solution is nonexplosive.
:::

:::{prf:proof}
Local Lipschitz SDE theory and interlacing at jump times construct the solution up to exits from compact sets. Stop at $\tau_R=\inf\{t:W(S_t)\geq R\}$; Dynkin's formula and Grönwall give $\mathbb E W(S_{t\wedge\tau_R})\leq e^{Ct}W(S_0)$. Thus $\mathbb P(\tau_R\leq t)\leq e^{Ct}W(S_0)/R\to0$. The assumed jump construction rules out accumulation of jump times. Killing subsequently stops the solution.
:::

(sec-gg-perturbation-analysis)=
## 3. Moment drift, weighted alignment, and recurrence

:::{div} feynman-prose
A force changes velocity immediately and position through subsequent transport. Keeping that distinction makes the moment calculation short. The diffusion contribution to a quadratic velocity moment is its trace; spatial derivatives of the diffusion do not appear in that backward-generator calculation.

Alignment requires a second distinction. Symmetric pair weights become asymmetric after each row is divided by its own degree. The conserved average for frozen positions is therefore degree-weighted. An estimate for the ordinary average must account for the change of norm.
:::

### 3.1. Exact component estimates

:::{prf:lemma} Frozen alignment dissipates degree-weighted variance
:label: lem-gg-viscous-dissipative

Hold $X$ fixed, let $D_X=\operatorname{diag}(d_i)$, and evolve $\dot V=-\nu L_XV$. Then

$$
\bar v_D=\frac{\sum_i d_iv_i}{\sum_i d_i}
$$

is constant, and

$$
\frac{d}{dt}\sum_i d_i|v_i-\bar v_D|^2
=-\nu\sum_{i,j}K_{ij}|v_i-v_j|^2\leq0.
$$

The matrix $\widetilde L_X=D_X^{1/2}L_XD_X^{-1/2}$ is symmetric with spectrum in $[0,2]$, and $\|L_X\|_{\mathrm{op}}\leq1+\sqrt{R_d}$.
:::

:::{prf:proof}
Symmetry gives $\sum_i d_i\dot v_i=\nu\sum_{i,j}K_{ij}(v_j-v_i)=0$. Differentiate the weighted variance and pair the $(i,j)$ and $(j,i)$ terms. The quadratic form of $D_X-K$ is $\tfrac12\sum_{i,j}K_{ij}|z_i-z_j|^2$, so the normalized similarity is positive semidefinite; $(z_i-z_j)^2\leq2(z_i^2+z_j^2)$ bounds its spectrum by two. The symmetric normalized adjacency has spectrum in $[-1,1]$, hence $\|W\|\leq\sqrt{R_d}$ by similarity and $\|I-W\|\leq1+\sqrt{R_d}$.
:::

:::{prf:remark} Moving degrees and ordinary momentum
:label: rem-gg-moving-alignment

For the alignment contribution alone, the degrees are frozen because that generator differentiates velocities. Along the full kinetic evolution, $d_i(X_t)$ changes and differentiating a weighted energy also produces $\dot d_i$ terms. Ordinary momentum satisfies
$\frac{d}{dt}\sum_i v_i=\nu\sum_{i,j}K_{ij}(v_j-v_i)/d_i$, which need not vanish. Strict weighted-variance decay requires a lower bound on the nonzero normalized-Laplacian eigenvalues; kernel smoothness alone supplies no such uniform gap.
:::

:::{prf:lemma} Diffusion and acceleration on quadratic moments
:label: lem-gg-diffusion-perturbation

Let $\bar v=N^{-1}\sum_i v_i$, $Q_v=N^{-1}\sum_i|v_i|^2$, and $V_v=Q_v-|\bar v|^2$. For independent particle noise blocks,

$$
L_{\mathrm{diff}}Q_v=\frac1N\sum_i\operatorname{tr}D_i,\qquad
L_{\mathrm{diff}}|\bar v|^2=\frac1{N^2}\sum_i\operatorname{tr}D_i,
$$

$$
L_{\mathrm{diff}}V_v=\frac{N-1}{N^2}\sum_i\operatorname{tr}D_i\leq d c_{\max}.
$$

An acceleration $F_i$ contributes

$$
L_FQ_v=\frac2N\sum_i v_i\cdot F_i,\qquad
L_FV_v=\frac2N\sum_i(v_i-\bar v)\cdot F_i.
$$

It contributes zero to a function of positions alone, including positional variance. If $|F_i|\leq A$, then $|L_FQ_v|\leq2A\sqrt{Q_v}$ and $|L_FV_v|\leq2A\sqrt{V_v}$.
:::

:::{prf:proof}
Apply $\tfrac12\sum_iD_i:\nabla_{v_i}^2$ and $\sum_iF_i\cdot\nabla_{v_i}$ to the displayed quadratics. The gradients of positional functions in velocity are zero. Cauchy–Schwarz gives the two force bounds. Subtracting the corresponding isotropic trace formulas gives the diffusion-perturbation estimate.
:::

(sec-gg-foster-lyapunov)=
### 3.2. A concrete coercive functional and perturbation transfer

:::{prf:definition} Quadratic hypocoercive moment functional
:label: def-gg-synergistic-lyapunov

Under the quadratic confinement assumptions, choose

$$
0<e\leq\min\{\gamma/2,\sqrt{m_U}/2\},
$$

and set

$$
W=1+\frac1N\sum_i\left[U(x_i)+\frac{|v_i|^2}{2}
+e x_i\cdot v_i+\frac{e\gamma}{2}|x_i|^2\right].
$$

Let $Q_x=N^{-1}\sum_i|x_i|^2$. Then

$$
1+\frac{m_U}{4}Q_x+\frac14Q_v\leq W
\leq1+C_W(Q_x+Q_v),
$$

where $C_W=\tfrac12\max\{M_U+e\gamma+e,1+e\}$.
:::

:::{prf:proof}
Use $m_U|x|^2/2\leq U(x)\leq M_U|x|^2/2$. For the lower bound, $e|x\cdot v|\leq m_U|x|^2/4+e^2|v|^2/m_U$, with $e^2/m_U\leq1/4$. For the upper bound, use $2|x\cdot v|\leq|x|^2+|v|^2$.
:::

:::{prf:theorem} Foster–Lyapunov transfer with explicit perturbation costs
:label: thm-gg-foster-lyapunov-drift

Let a backbone backward generator satisfy $L_0W\leq-\kappa_0W+C_0$, and let every added term have a proved bound

$$
L_jW\leq a_jW+b_j.
$$

If $\kappa=\kappa_0-\sum_j a_j>0$, then

$$
LW\leq-\kappa W+C,\qquad C=C_0+\sum_jb_j.
$$

The constants are uniform in $N$ when the input constants are. For a discrete update, the same conclusion follows from bounds on the actual kernel difference $(P-P_0)W$; continuous generators cannot be substituted for that difference without a step-error estimate.
:::

:::{prf:proof}
Sum the pointwise backward-generator inequalities. In discrete time, sum $P_0W\leq q_0W+C_0$ and $(P-P_0)W\leq aW+b$, obtaining $PW\leq(q_0+a)W+C_0+b$ when $q_0+a<1$.
:::

:::{prf:proposition} Direct drift for bounded adaptive forces
:label: prop-gg-explicit-conservative-drift

Use $W$ above and position-dependent noise with $D_i\preceq c_{\max}I$. Suppose $|F_i|\leq A$ and $R_d\leq R_*$. Put

$$
a_v=\gamma-e,\qquad a_x=e m_U,\qquad C_L=1+\sqrt{R_*}.
$$

If

$$
\nu C_L(1+e/2)\leq a_v/4,
\qquad \nu C_Le/2\leq a_x/4,
$$

then the conservative diffusion satisfies

$$
LW\leq-\kappa W+C,
\qquad \kappa=\frac{\min\{a_x,a_v\}}{2C_W},
$$

$$
C=\kappa+\frac{d c_{\max}}2
+A^2\left(\frac1{a_v}+\frac{e^2}{a_x}\right).
$$

An additional cloning kernel with $JW\leq a_JW+b_J$ preserves this drift when $a_J<\kappa$, with the constants from the preceding theorem.
:::

:::{prf:proof}
Transport cancels the force contribution to $U+|v|^2/2$, and the cross terms cancel the $x\cdot v$ friction term. Thus the unperturbed deterministic contribution is

$$
-\frac1N\sum_i[(\gamma-e)|v_i|^2+e x_i\cdot\nabla U(x_i)]
\leq-a_vQ_v-a_xQ_x.
$$

The diffusion contributes at most $dc_{\max}/2$. For the bounded force,

$$
\frac1N\sum_i(v_i+e x_i)\cdot F_i
\leq\frac{a_v}4Q_v+\frac{a_x}4Q_x
+A^2\left(\frac1{a_v}+\frac{e^2}{a_x}\right).
$$

The alignment contribution has absolute value at most
$\nu C_L[Q_v+e\sqrt{Q_xQ_v}]$.
Use $2\sqrt{Q_xQ_v}\leq Q_x+Q_v$ and the two displayed parameter inequalities to bound it by $a_vQ_v/4+a_xQ_x/4$. The remaining negative drift is at least half of each original quadratic coefficient. Finally $W-1\leq C_W(Q_x+Q_v)$ yields the claimed constants.
:::

:::{div} feynman-prose
This direct estimate shows that a bounded adaptive force changes the moment bound through an additive constant. The sufficient alignment bounds arise because we estimated its contribution in an unweighted norm. Weighted estimates can improve them when the motion of the degrees is controlled.

Failure of one sufficient inequality means that this estimate no longer closes. It does not prove that the dynamics diverges. Population-independent moment constants also do not, by themselves, give population-independent mixing constants.
:::

(sec-support-thm-backbone-convergence)=
:::{prf:corollary} Backbone moment drift
:label: proof-thm-backbone-convergence

For $F_i=0$, $\nu=0$, and constant kinetic noise, the preceding calculation gives a coercive kinetic drift with constants independent of $N$. For the discrete backbone with cloning, the complete component-and-composition proof is {doc}`06_convergence`, using the exact cloning estimates in {doc}`03_cloning` and kinetic estimates in {doc}`05_kinetic_contraction`. Applying a geometric perturbation uses the same functional and actual operator difference in {prf:ref}`thm-gg-foster-lyapunov-drift`.
:::

:::{prf:proof}
Set the adaptive and alignment terms to zero in the explicit generator calculation. For a discrete composition, the referenced component inequalities apply successively to the selected Lyapunov function; the expectation under the first substep must be retained when estimating the next. This gives the stated backbone input, whose perturbation is the preceding theorem.
:::

(sec-support-cor-exp-convergence)=
:::{prf:corollary} Iteration of a discrete Lyapunov bound
:label: proof-cor-exp-convergence

If $PW\leq qW+C$, $0<q<1$, then

$$
\mathbb E W(S_n)\leq q^n\mathbb E W(S_0)+\frac{C}{1-q}(1-q^n).
$$

This is an upper envelope for the moment; its limiting level need not equal the stationary expectation.
:::

:::{prf:proof}
Take total expectations and iterate $w_{n+1}\leq qw_n+C$. Summing the geometric series gives the formula.
:::

(sec-gg-ergodicity)=
### 3.3. Smoothing, minorization, and the target law

(sec-support-lem-hormander)=
:::{prf:lemma} Kinetic Hörmander brackets
:label: proof-lem-hormander

For smooth kinetic coefficients, suppose the velocity noise matrix is invertible. The velocity noise fields together with their brackets with the kinetic drift span the full phase-space tangent space. For constant isotropic noise,

$$
X_j=\sigma\partial_{v_j},\qquad
X_0=v\cdot\nabla_x-\nabla U\cdot\nabla_v-\gamma v\cdot\nabla_v,
$$

and $[X_0,X_j]=-\sigma\partial_{x_j}+\gamma X_j$.
:::

:::{prf:proof}
The noise fields span velocity directions. For a general velocity noise column $B_j=(0,\Sigma_{:j})$, the position component of $[X_0,B_j]$ is $-\Sigma_{:j}$, because $\partial_v(v\cdot\nabla_x)$ is the identity. Invertibility therefore supplies every position direction modulo velocity directions. The displayed constant-noise bracket follows by direct differentiation. With smooth coefficients and the usual local domain conditions, Hörmander's theorem gives local hypoelliptic regularity. It does not give uniform global density bounds or uniform-in-$N$ minorization constants.
:::

:::{prf:lemma} Minorization from a positive transition density
:label: lem-gg-phi-irreducibility

For a specified conservative transition kernel $P_T$, suppose its density $p_T(S,S')$ is continuous and positive on $C\times B$, where $C,B$ are compact and $B$ has positive reference volume. Then

$$
P_T(S,\cdot)\geq\varepsilon\varphi(\cdot),\quad S\in C,
\qquad \varphi=\frac{1_B\,dS'}{|B|},
\quad \varepsilon=|B|\inf_{C\times B}p_T>0.
$$

If the chain reaches $C$ from every initial state and the target sets are accessible through such density neighborhoods, it is irreducible for the corresponding reference measure.
:::

:::{prf:proof}
Continuity and compactness make the positive density infimum strictly positive. Integrate the lower bound over a measurable target set. Concatenating this step with a positive-probability path into $C$ proves the stated accessibility implication. A killed kernel requires survival along the path and the appropriate killed-process criterion.
:::

:::{prf:lemma} A sufficient aperiodicity condition
:label: lem-gg-aperiodicity

An irreducible discrete skeleton is strongly aperiodic if it has a one-step small set $C$ with $P(S,\cdot)\geq\varepsilon\varphi(\cdot)$ for $S\in C$ and $\varphi(C)>0$.
:::

:::{prf:proof}
The minorization permits a return from the small set to itself in one skeleton step with positive mass. This excludes a cyclic decomposition with period greater than one. A positive density on $C\times C$ supplies the condition. Absolute continuity alone does not exclude periodicity.
:::

:::{prf:theorem} Harris convergence for a conservative geometric process
:label: thm-gg-geometric-ergodicity

Let $P$ be a conservative skeleton with $PW\leq qW+C$, $q<1$. Suppose a sufficiently large level set of $W$ has the preceding minorization and aperiodicity properties. Then $P$ has a unique invariant probability $\pi_N$ with finite $W$ moment, and

$$
\|P^n(S,\cdot)-\pi_N\|_{\mathrm{TV}}
\leq M_NW(S)r_N^n,\qquad r_N<1.
$$

Uniformity of $r_N$ requires uniform drift and minorization data. For a killed process, the conditional law is $\mu Q_t/\mu Q_t1$; existence and convergence of its QSD use the killed-process hypotheses in {doc}`06_convergence`, or the Doob and entropy routes in {doc}`15_kl_convergence`.
:::

:::{prf:proof}
The drift forces repeated returns to a bounded level set. On that set, minorization supplies a fixed probability of coupling two copies; between visits, the drift bounds the weighted cost. The Harris theorem makes this construction quantitative in weighted total variation; see [Hairer and Mattingly](https://www.hairer.org/papers/harris.pdf). Its constants depend on both pieces of data. The theorem concerns a conservative kernel, so its stationary law is invariant; the separately displayed normalization specifies the killed law.
:::

(sec-gg-lsi)=
## 4. Static LSI and dynamical entropy estimates

:::{div} feynman-prose
An LSI is an inequality for a probability law. Entropy dissipation is an identity for an evolution. They work together when the derivative controls the same information that appears in the inequality. For kinetic noise, the immediate dissipation sees velocity derivatives; the modified entropy also sees position derivatives.

The complete constant-diffusion calculation is already available in chapter 15. We retain it as a proved reference result and write the extra geometric terms explicitly. This prevents a coefficient regularity estimate from being mistaken for a functional inequality for an unidentified stationary law.
:::

(sec-gg-lsi-microscopic)=
### 4.1. Three information quantities

:::{prf:definition} Velocity dissipation and full modified Fisher information
:label: def-gg-hypocoercive-fisher

For a specified continuous joint law $\pi_N$ and relative density $h$, define

$$
I_x=\int h\sum_i|\nabla_{x_i}\log h|^2d\pi_N,\quad
I_v=\int h\sum_i|\nabla_{v_i}\log h|^2d\pi_N,\quad I=I_x+I_v,
$$

$$
I_v^D=\int h\sum_i(\nabla_{v_i}\log h)^{\mathsf T}D_i\nabla_{v_i}\log h\,d\pi_N.
$$

Then $c_{\min}I_v\leq I_v^D\leq c_{\max}I_v$. For a constant positive definite full phase-space matrix $G$, set

$$
\Phi_G(h)=H_{\pi_N}(h)+I_G(h),\qquad
I_G(h)=\int h(\nabla\log h)^{\mathsf T}G\nabla\log h\,d\pi_N.
$$

The velocity information $I_v^D$ and full information $I_G$ have different domains of coercivity.
:::

:::{prf:lemma} Entropy production for the actual law
:label: lem-gg-velocity-fisher-dissipation

For a conservative diffusion-jump generator preserving $\pi_N$, with velocity covariance blocks $D_i$, the density evolution satisfies

$$
\frac{d}{dt}H_{\pi_N}(h_t)
=-\frac12 I_v^D(h_t)-\mathcal D_J(h_t),
$$

where

$$
\mathcal D_J(h)=\iint
\left[h(S)\log\frac{h(S)}{h(S')}-h(S)+h(S')\right]
\pi_N(dS)r_N(S,dS')\geq0.
$$

For an actual QSD $\nu_N$ with interior killing rate $\kappa$, on a domain without outgoing boundary flux,

$$
\dot H_{\nu_N}=-\frac12I_v^D-\mathcal D_J
+f_t(\kappa)H_{\nu_N}-\nu_N[\kappa(h\log h-h+1)].
$$
:::

:::{prf:proof}
Apply the diffusion chain rule and the jump Bregman identity in {prf:ref}`prop-kl-conditioned-entropy`, with $a=\tfrac12\operatorname{diag}(0,D_i)$. For an invariant conservative law, the eigenvalue and killing terms vanish. For the QSD, use $L^*\nu_N=(\kappa-\lambda_N)\nu_N$ and the survival-normalized equation. Absorbing boundaries add the outgoing flux and total-loss normalization specified in {prf:ref}`prop-kl-boundary-entropy`.
:::

:::{prf:remark} Spatial tests and friction
:label: rem-gg-velocity-lsi-obstruction

For $f=f(X)$ nonconstant under the position marginal, $\int\sum_i|\Sigma_i\nabla_{v_i}f|^2d\pi_N=0$ while $\operatorname{Ent}_{\pi_N}(f^2)>0$. Thus an LSI for the full law cannot use only this velocity form. Friction alone is a first-order drift; the displayed entropy dissipation belongs to the complete generator relative to its invariant law, or to the normalized QSD identity.
:::

(sec-gg-lsi-macroscopic)=
### 4.2. The extra derivatives generated by geometry

:::{prf:lemma} Exact transport-diffusion commutator
:label: lem-gg-commutator-expansion

Let $T=v\cdot\nabla_x$ and $A=D(x,v):\nabla_v^2$, with a symmetric matrix $D$. Then

$$
[T,A]f=(v\cdot\nabla_xD):\nabla_v^2f
-2\sum_{a,b}D_{ab}\partial_{x_a}\partial_{v_b}f.
$$
:::

:::{prf:proof}
Expand $T(D_{ab}\partial_{v_a}\partial_{v_b}f)$. In the opposite composition, differentiating $v_c\partial_{x_c}f$ twice in velocity produces its transported second derivative and the two mixed derivatives $\partial_{x_a}\partial_{v_b}f$ and $\partial_{x_b}\partial_{v_a}f$. Symmetry combines them into the displayed factor two. The mixed term remains even when $D$ is constant.
:::

:::{prf:lemma} A weighted commutator estimate
:label: lem-gg-commutator-error

For a term in a Fisher calculation bounded by

$$
|E(h)|\leq K_D\int h|v|\,|q|\,|\nabla_vq|\,d\pi_N,
\qquad q=\nabla\log h,
$$

one has, for every $\varepsilon>0$,

$$
|E(h)|\leq\varepsilon\int h|\nabla_vq|^2d\pi_N
+\frac{K_D^2}{4\varepsilon}\int h|v|^2|q|^2d\pi_N.
$$

The tensor contraction defining $K_D$ must use the full swarm derivative norm needed by the calculation.
:::

:::{prf:proof}
Apply $ab\leq\varepsilon a^2+b^2/(4\varepsilon)$ pointwise, with $a=|\nabla_vq|$ and $b=K_D|v||q|$, then integrate. A bound on $\pi_N(|v|^2)$ alone does not control the last integral, because its weight is $h|q|^2$.
:::

:::{prf:lemma} Retaining the second-derivative dissipation
:label: lem-gg-velocity-second-derivative

For the constant-diffusion Gibbs reference of {prf:ref}`lem-kinetic-evolution-bounds`, the full Fisher derivative contains

$$
-2D_0\int h\sum_j
(\partial_{v_j}q)^{\mathsf T}G(\partial_{v_j}q)d\pi_N
\leq-2D_0\lambda_{\min}(G)\int h|\nabla_vq|^2d\pi_N.
$$

This term absorbs a commutator contribution of the preceding form when its coefficient is smaller than the displayed second-derivative coefficient; the remaining weighted Fisher term must still be estimated.
:::

:::{prf:proof}
The exact matrix identity in {prf:ref}`lem-kinetic-evolution-bounds` gives the first expression. Apply positive definiteness of $G$ and subtract the chosen Young coefficient. A bound of arbitrary second derivatives by first-order Fisher information cannot replace this calculation: even on a bounded smooth domain, oscillatory functions have second derivatives growing faster than their first derivatives.
:::

(sec-support-lem-macro-transport)=
:::{prf:lemma} Macroscopic transport from a position Poincaré inequality
:label: proof-lem-macro-transport

Let $\pi(dx,dv)=\pi_x(dx)\pi(dv\mid x)$. Suppose $\pi_x$ has Poincaré constant $1/\kappa_x$ and its conditional second-moment matrix satisfies

$$
M_v(x):=\int vv^{\mathsf T}\pi(dv\mid x)\succeq c_vI.
$$

For every mean-zero $a\in H^1(\pi_x)$,

$$
\|a\|_{L^2(\pi_x)}^2
\leq\frac1{\kappa_xc_v}\|v\cdot\nabla_xa\|_{L^2(\pi)}^2.
$$

This applies to a velocity average $a=\Pi h-\pi(h)$ when that average belongs to $H^1(\pi_x)$. Centered conditional velocities make $M_v$ their covariance; centering is unnecessary if the displayed second-moment bound is known directly.
:::

:::{prf:proof}
Poincaré gives $\|a\|^2\leq\kappa_x^{-1}\int|\nabla a|^2d\pi_x$. At each $x$,
$\int|v\cdot\nabla a|^2\pi(dv\mid x)=\nabla a^{\mathsf T}M_v(x)\nabla a\geq c_v|\nabla a|^2$. Integrate and combine.
:::

(sec-gg-lsi-gap)=
### 4.3. Full-gradient LSI and the dynamical gap

:::{prf:theorem} N-uniform LSI for an identified geometric law
:label: thm-gg-lsi-main

Let $\pi_N$ be a specified continuous joint invariant law or QSD satisfying one of the proved structural criteria in {prf:ref}`cor-n-uniform-lsi`: product kinetic reference, bounded joint density tilt, uniform joint curvature, or contractive additive-noise invariant flow. Write its uniform full-gradient LSI constant as $C_*$. Then

$$
\operatorname{Ent}_{\pi_N}(f^2)
\leq2C_*\int\sum_i(|\nabla_{x_i}f|^2+|\nabla_{v_i}f|^2)d\pi_N.
$$

For the full geometric comparison matrix $\mathsf A_N=\operatorname{diag}(I,D_1,\ldots,D_N)$, with the position and velocity coordinates ordered consistently,

$$
\operatorname{Ent}_{\pi_N}(f^2)
\leq\frac{2C_*}{\min\{1,c_{\min}\}}
\int(\nabla f)^{\mathsf T}\mathsf A_N\nabla f\,d\pi_N.
$$

If instead $a\leq d\widetilde\pi_N/d\pi_N\leq b$ uniformly, the constant for $\widetilde\pi_N$ is at most $(b/a)C_*$ before geometric form comparison.
:::

:::{prf:proof}
The four structural routes and their full proofs give the first inequality. Velocity ellipticity gives $\mathsf A_N\succeq\min\{1,c_{\min}\}I$ and therefore the second. Bounded density comparison is {prf:ref}`thm-lsi-perturbation`. This comparison concerns densities; a bound on an adaptive drift alone is not a density-ratio bound.
:::

:::{prf:proposition} Entropy-Fisher closure for the complete geometric evolution
:label: prop-gg-entropy-fisher-gap

Let the actual law $\pi_N$ satisfy the preceding LSI, let $G_N\succ0$ with $G_N\preceq g_+I$, and suppose the complete normalized derivative satisfies

$$
\frac{d}{dt}\Phi_{G_N}(h_t)\leq-\delta_N I(h_t),
\qquad\delta_N>0.
$$

Then

$$
\Phi_{G_N}(h_t)\leq
\exp\!\left[-\frac{\delta_Nt}{C_*/2+g_+}\right]\Phi_{G_N}(h_0).
$$

The derivative includes coefficient derivatives, viscous and cloning terms, and survival or boundary normalization when present.
:::

:::{prf:proof}
LSI gives $H\leq C_*I/2$ and $I_{G_N}\leq g_+I$. Substitute $\Phi_{G_N}\leq(C_*/2+g_+)I$ into the derivative bound and apply Grönwall. This is the actual-law theorem {prf:ref}`thm-kl-convergence-euclidean`. The exact first variation for checking its premise is {prf:ref}`lem-kl-functional-first-variation`.
:::

:::{prf:corollary} Separate moment and entropy margins
:label: cor-gg-joint-thresholds

If the moment estimates give $\kappa_0-\sum_j a_j>0$, and the full entropy calculation gives $\delta_N\geq\delta_*>0$ with $C_*,g_+$ uniform, then the moment drift and entropy rate are both uniform. In a same-reference perturbation calculation of the form

$$
\dot\Phi_G\leq-[\eta-\epsilon_F A_F-A_D-\nu A_V-A_J-A_\kappa]I,
$$

a sufficient entropy condition is positivity of the displayed bracket. Each coefficient must come from the actual derivative estimate; persistent additive or square-root forcing produces the floor in {prf:ref}`lem-hypocoercive-forcing-floor`.
:::

:::{prf:proof}
Apply {prf:ref}`thm-gg-foster-lyapunov-drift` and {prf:ref}`prop-gg-entropy-fisher-gap` to their respective margins. No inequality comparing moment drift to an entropy derivative is used.
:::

:::{div} feynman-prose
Several complete analytic regimes are available. The nonconvex kinetic Gibbs law has a proved LSI and an explicit hypocoercive rate. A common-invariant cloning kernel with controlled Fisher amplification preserves that rate with a quantified loss. A conservative frozen alignment field has a non-Gibbs stationary LSI from the flow-contraction proof in {prf:ref}`cor-kl-frozen-alignment-lsi`. Frozen row-normalized velocity dynamics has the Gaussian covariance and LSI bound in {prf:ref}`prop-kl-frozen-ou-lsi`.

The moving, cloning QSD needs its own law identification and complete derivative estimate. Uniform velocity ellipticity and bounded coefficient derivatives are inputs to that calculation. They do not erase the weighted Fisher term or make the frozen Gaussian law its conditional velocity law.
:::

(sec-support-cor-exponential-qsd-companion-dependent-full)=
:::{prf:corollary} L2 decay when the generator form matches the inequality
:label: proof-cor-exponential-qsd-companion-dependent-full

For a conservative invariant semigroup with relative-density generator $K$, suppose

$$
-\langle g,Kg\rangle_{L^2(\pi)}\geq c\|g\|_{L^2(\pi)}^2,
\qquad \pi(g)=0.
$$

Then $\|h_t-1\|_{L^2(\pi)}\leq e^{-ct}\|h_0-1\|_{L^2(\pi)}$. A full-gradient Poincaré inequality yields this premise when the actual Dirichlet form dominates that full gradient. For a kinetic generator with velocity-only diffusion, use a proved hypocoercive norm estimate instead. A QSD requires the normalized or Doob-transformed generator.
:::

:::{prf:proof}
Differentiate the squared norm: $\frac d{dt}\|h_t-1\|^2=2\langle h_t-1,K(h_t-1)\rangle\leq-2c\|h_t-1\|^2$. Integrate. The form-matching condition is exactly the step needed to apply Poincaré; a position-only test explains its failure for velocity-only diffusion.
:::

(sec-gg-mean-field-lsi)=
## 5. Mean-field laws, concentration, and tails

:::{div} feynman-prose
The empirical measurement formulas extend to a probability measure by replacing sums with integrals and then performing the same normalization. Nonlinear operations retain their order. We first average the measurements, form the variance and Z-score, and only then apply the chosen fitness map.

Existence of the mean-field equation, convergence of particle marginals, and an LSI for a stationary limit are separate steps. The following statements connect them without replacing an empirical measure by a smooth density or assuming a dimension-independent empirical Wasserstein rate.
:::

### 5.1. The nonlinear generator and particle comparison

:::{prf:definition} Geometric mean-field coefficients
:label: def-gg-mean-field-generator

For a probability law $\mu$ on phase space, set

$$
\mu_\rho[d](x)=\frac{\int K_\rho(x,y)d(y)\mu(dy,dw)}
{\int K_\rho(x,y)\mu(dy,dw)},
$$

and form the localized variance, regularized Z-scores, and $V_{\mathrm{fit}}[\mu]$ in the same order as the finite-swarm definitions. The frozen alignment mean is

$$
a_\mu(x)=\frac{\int K(x,y)w\,\mu(dy,dw)}{\int K(x,y)\mu(dy,dw)}.
$$

The conservative Itô generator is

$$
\begin{aligned}
L_\mu\phi={}&v\cdot\nabla_x\phi+
[-\nabla U+\epsilon_F\nabla V_{\mathrm{fit}}[\mu]-\gamma v
+\nu(a_\mu-v)+b_{\mathrm{geo}}[\mu]]\cdot\nabla_v\phi\\
&+\frac12D[\mu]:\nabla_v^2\phi+J_\mu\phi.
\end{aligned}
$$

Here $J_\mu$ must be derived from the actual companion and cloning rule. The weak equation is $\frac d{dt}\mu_t(\phi)=\mu_t(L_{\mu_t}\phi)$; killing adds the corresponding loss and survival-normalization terms. In general $V_{\mathrm{fit}}[\mu]\ne\int V_{\mathrm{fit}}[\delta_z]d\mu(z)$.
:::

:::{prf:proposition} A finite-time synchronous particle comparison
:label: prop-gg-propagation-chaos

Suppose an identified nonlinear diffusion has globally Lipschitz state coefficients and a particle coupling for which, writing $Z_i^N$ and independent nonlinear copies $\bar Z_i$,

$$
e_N(t):=\frac1N\sum_i\mathbb E|Z_i^N(t)-\bar Z_i(t)|^2
$$

satisfies the coefficient bounds needed for

$$
e_N'(t)\leq C e_N(t)+r_N(t).
$$

Then

$$
e_N(t)\leq e^{Ct}e_N(0)+\int_0^t e^{C(t-s)}r_N(s)ds.
$$

For exchangeable couplings, the squared $W_2$ distance between $k$-particle marginals is at most $k e_N(t)$. For empirical measures,

$$
\mathbb E W_2^2(\mu_N(t),\mu_t)
\leq2e_N(t)+2\mathbb E W_2^2\!\left(\frac1N\sum_i\delta_{\bar Z_i(t)},\mu_t\right).
$$

If $e_N(0)=0$ and $r_N\leq C_T/N$, the coupled particle error is $O_T(N^{-1})$. The remaining empirical sampling term retains its dimension and moment dependence.
:::

:::{prf:proof}
For a synchronous diffusion coupling, Itô's formula gives drift inner products and the squared Hilbert–Schmidt difference of diffusion coefficients. Split each coefficient difference into the change of particle state and the discrepancy of the empirical field built from the independent copies. Lipschitz and Young inequalities give the stated differential bound, with the latter discrepancy recorded as $r_N$. Integrate by Grönwall. The first $k$ coupled coordinates give the marginal transport bound; matching particle labels gives the empirical coupling cost, and the squared triangle inequality gives the last display. A jump coupling requires its own contribution to the same differential estimate.
:::

:::{prf:remark} Well-posedness and the actual coefficient class
:label: rem-gg-mean-field-well-posedness

For globally Lipschitz coefficients in state and $W_2$-law, with linear growth, the conservative McKean–Vlasov diffusion is well posed for finite-second-moment initial data: Picard iteration on law paths is a contraction on a sufficiently short interval by the same synchronous estimate, and moment bounds permit iteration in time. The normalized coefficient calculus determines when the Geometric Gas belongs to that class. The more specific weighted-space and cloning arguments are developed in {doc}`08_mean_field` and {doc}`09_propagation_chaos`; their precise hypotheses and conclusions apply to the corresponding identified equation.
:::

:::{prf:theorem} LSI passes to a stationary marginal limit
:label: thm-gg-mean-field-lsi

Suppose the specified joint laws $\pi_N$ satisfy {prf:ref}`thm-gg-lsi-main` with uniform Euclidean constant $C_*$. Every fixed marginal has the same constant. If those marginals converge weakly to $\pi^{(k)}$, then

$$
\operatorname{Ent}_{\pi^{(k)}}(f^2)\leq2C_*\int|\nabla f|^2d\pi^{(k)}
$$

for bounded smooth tests with bounded continuous squared gradient, and on their Sobolev closure when they form a core. If the mean-field existence, uniqueness, and stationary-limit arguments identify $\pi^{(1)}$ with a stationary solution, that stationary mean-field law has LSI constant at most $C_*$.
:::

:::{prf:proof}
Apply the joint inequality to a test depending on only $k$ coordinates. All other derivatives vanish. Weak convergence passes both bounded continuous integrands to the limit; closure extends the form domain. This is the proof of {prf:ref}`cor-kl-lsi-mean-field-limit`. The stationary identification is supplied by the mean-field theorem for the equation, not by tensorization.
:::

(sec-gg-implications)=
### 5.2. Entropy convergence and concentration

:::{prf:corollary} KL convergence for the controlled full evolution
:label: cor-gg-kl-convergence

Under {prf:ref}`prop-gg-entropy-fisher-gap`,

$$
D_{\mathrm{KL}}(\mu_t\Vert\pi_N)
\leq e^{-rt}\Phi_{G_N}(d\mu_0/d\pi_N),
\qquad r=\frac{\delta_N}{C_*/2+g_+}.
$$

This concerns the law of the swarm. Its empirical atomic measure generally has infinite KL relative to a positive continuous density.
:::

:::{prf:proof}
Use $H\leq\Phi_{G_N}$ in the modified-entropy bound. The initial modified entropy must be finite, or the argument starts from a positive time when it is finite.
:::

:::{prf:corollary} Concentration from the joint LSI
:label: cor-gg-concentration

For the actual law satisfying the Euclidean LSI with constant $C_*$ and an $L$-Lipschitz observable $F$,

$$
\pi_N(|F-\pi_NF|\geq r)\leq2\exp[-r^2/(2C_*L^2)].
$$

In particular, the empirical average of a one-particle $L$-Lipschitz observable has variance at most $C_*L^2/N$.
:::

:::{prf:proof}
For bounded $F$, let $M(t)=\pi_N(e^{tF})$. Applying LSI to $e^{tF/2}$ gives
$tM'(t)-M(t)\log M(t)\leq C_*L^2t^2M(t)/2$.
Divide by $t^2M(t)$ and integrate from zero to obtain
$\log\pi_N(e^{t(F-\pi_NF)})\leq C_*L^2t^2/2$.
Chernoff's bound, optimized over $t$, gives each tail. Truncation extends the result to Lipschitz $F$. Linearizing LSI gives Poincaré; the empirical-average gradient has squared norm at most $L^2/N$.
:::

(sec-support-thm-exponential-tails)=
:::{prf:theorem} Exponential moment and tail bounds
:label: proof-thm-exponential-tails

For a conservative diffusion with backward generator $L$, suppose a coercive $V\geq0$ satisfies

$$
LV\leq-\beta V+C,\qquad
\Gamma_L(V,V)\leq aV+b,
\qquad\beta>0.
$$

For $\theta>0$ sufficiently small, $W_\theta=e^{\theta V}$ satisfies an exponential Lyapunov bound $LW_\theta\leq-\eta W_\theta+B_\theta$ for some finite constants. An invariant law in the corresponding integrability domain then satisfies

$$
\pi(V\geq R)\leq\frac{B_\theta}{\eta}e^{-\theta R}.
$$

For a killed generator $A=L-\kappa$, if the full bound $AW_\theta\leq-\eta W_\theta+B_\theta$ holds and a QSD has eigenvalue $\lambda<\eta$, then

$$
\nu(W_\theta)\leq\frac{B_\theta}{\eta-\lambda},
\qquad
\nu(V\geq R)\leq\frac{B_\theta}{\eta-\lambda}e^{-\theta R}.
$$

Use cutoff/domain justification when integrating unbounded functions. Jump processes require their actual exponential jump contribution in the bound.
:::

:::{prf:proof}
The diffusion chain rule gives

$$
LW_\theta=W_\theta[\theta LV+\theta^2\Gamma_L(V,V)]
\leq\theta W_\theta[-(\beta-\theta a)V+C+\theta b].
$$

Choose $\beta-\theta a>0$. Outside a sufficiently large level set this is at most $-\eta W_\theta$; on that level set absorb the remaining bound into $B_\theta$. Integrating against an invariant law gives $\eta\pi(W_\theta)\leq B_\theta$. For a QSD, use $\nu(AW_\theta)=-\lambda\nu(W_\theta)$, giving $(\eta-\lambda)\nu(W_\theta)\leq B_\theta$. Markov's inequality gives both tails. These are probability and moment estimates; a pointwise Gaussian density bound requires an additional quantitative density estimate.
:::

(sec-support-prop-complete-gradient-bounds)=
:::{prf:proposition} Local logarithmic density derivative bounds
:label: proof-prop-complete-gradient-bounds

If a density $p$ is strictly positive and $C^2$ on a neighborhood of a compact interior set $K$, then

$$
\|\nabla_x\log p\|_{L^\infty(K)}<\infty,
\qquad\|\Delta_v\log p\|_{L^\infty(K)}<\infty.
$$

The assertion is local and does not assume that $p$ is supported on $K$.
:::

:::{prf:proof}
On $K$, positivity and compactness give $m_K=\inf_Kp>0$, while the first two derivatives are bounded. Use
$\nabla\log p=\nabla p/p$ and $\Delta\log p=\Delta p/p-|\nabla p|^2/p^2$.
The bounds may deteriorate as $K$ expands or approaches a killing boundary.
:::

(sec-support-lem-variance-to-gap-adaptive)=
:::{prf:lemma} Variance gives a deviation in the support
:label: proof-lem-variance-to-gap-adaptive

If a real random variable has mean $\mu$ and variance $s^2>0$, then
$\sup_{x\in\operatorname{supp}X}|x-\mu|\geq s$. For bounded support the supremum is attained.
:::

:::{prf:proof}
Writing the supremum as $R\in[0,\infty]$, one has $|X-\mu|\leq R$ almost surely, so $s^2\leq R^2$. A bounded support is closed and hence compact; continuity attains the supremum. This is a statement about measurement spread, not a Markov-generator spectral gap.
:::

:::{prf:conjecture} Pairwise WFR contraction
:label: conj-gg-wfr-contraction

For a specified geometric evolution and a specified Wasserstein–Fisher–Rao metric, a possible stronger property is

$$
\mathrm{WFR}(\mathcal T_t\mu,\mathcal T_t\nu)
\leq e^{-ct}\mathrm{WFR}(\mu,\nu),\qquad c>0.
$$

Such a pairwise contraction requires a coupling or metric-evolution estimate for that operator. Entropy convergence to a target, diffusion-metric duality, and a cloning reweighting rule do not by themselves establish this property. The law-based transport consequences of the proved entropy bounds are developed in {doc}`11_hk_convergence`.
:::

(sec-gg-appendix-c)=
## 6. Holonomy, expansion, and moving-cell geometry

:::{div} feynman-prose
Once a positive metric is specified, its connection and curvature are ordinary differential-geometric objects. They describe how vectors rotate around a loop and how nearby trajectories expand or focus. These calculations do not require a probabilistic convergence theorem.

A diffusion coefficient can supply a metric, but the remaining dynamics must still be identified. In particular, velocity diffusion with covariance $g^{-1}$ is not the Laplace–Beltrami operator on position space. A Riemannian metric is also positive definite; a Lorentzian spacetime interpretation requires a separate construction.
:::

:::{prf:definition} Connection and curvature of the regularized metric
:label: def-gg-metric-connection

On a fixed coordinate domain where $g=H+\epsilon_\Sigma I\succ0$, its Levi-Civita connection is

$$
\Gamma^a_{bc}=\frac12g^{ad}(\partial_bg_{cd}+\partial_cg_{bd}-\partial_dg_{bc}).
$$

For $g\in C^2$, its curvature is

$$
R^a{}_{bcd}=\partial_c\Gamma^a_{db}-\partial_d\Gamma^a_{cb}
+\Gamma^a_{ce}\Gamma^e_{db}-\Gamma^a_{de}\Gamma^e_{cb}.
$$

A $C^3$ fitness potential supplies a $C^1$ Hessian metric and continuous connection; the ordinary $C^2$ metric curvature calculation is justified, for example, by a $C^4$ potential. The Laplace–Beltrami operator is

$$
\Delta_g f=|g|^{-1/2}\partial_a(|g|^{1/2}g^{ab}\partial_bf),
$$

including its first-order terms. These definitions do not impose an Einstein equation on $g$.
:::

(sec-appx-geometric-gas-holonomy)=
### 6.1. Curvature and small-loop holonomy

:::{prf:theorem} Ambrose–Singer holonomy theorem
:label: appx-ambrose-singer

For a connected smooth Riemannian manifold and its Levi-Civita connection, the Lie algebra of the restricted holonomy group at $p$ is spanned by curvature endomorphisms transported back from points reached by piecewise smooth paths:

$$
\mathfrak{hol}_p
=\operatorname{span}\{P_\gamma^{-1}R_q(X,Y)P_\gamma:
\gamma:p\to q,\ X,Y\in T_qM\}.
$$
:::

:::{prf:proof}
This is the classical holonomy theorem: curvature is the vertical component of the commutator of horizontal lifts, so infinitesimal horizontal loops generate the transported curvature endomorphisms. Conversely, horizontal transport preserves the distribution generated by those curvature directions, restricting the infinitesimal holonomy to their span. The bundle argument is given in {cite}`ambrose1953theorem,kobayashi1963foundations`.
:::

:::{prf:lemma} Shape-controlled small-loop expansion
:label: appx-holonomy-small-loops

Let $g\in C^3$ on a fixed normal neighborhood. Consider a small coordinate rectangle based at $p$, with orthonormal initial directions $X,Y$, side lengths $r,s$, and $\ell=\max\{r,s\}$, where $\min\{r,s\}\geq c\ell$ for fixed $c>0$. Choose the orientation corresponding to the sign below. If $K=\sup|R|$ and $K_1=\sup|\nabla R|$, then

$$
\mathrm{Hol}_\gamma V=V+rsR_p(X,Y)V+E,
\qquad
|E|\leq C(K_1\ell^3+K^2\ell^4)|V|.
$$

Thus for a shape-controlled family with area $A\asymp\ell^2$, the remainder is $O(K_1A^{3/2}+K^2A^2)$.
:::

:::{prf:proof}
Choose a frame obtained by radial parallel transport from $p$. Its connection one-form is $O(K\ell)$ on the rectangle, and its curvature differs from the transported $R_p$ by $O(K_1\ell+K^2\ell^2)$. Expand the parallel-transport integral equation around the loop once. The first curvature integral is $rsR_p(X,Y)$; curvature variation contributes $O(K_1\ell^3+K^2\ell^4)$. The second and higher ordered connection integrals contribute $O((K\ell^2)^2)$, after shrinking the neighborhood so the integral equation is uniformly controlled. These bounds give the stated remainder. The $K^2A^2$ term remains even for parallel curvature, and the diameter control is needed to express the error solely through area.
:::

(sec-appx-geometric-gas-raychaudhuri)=
### 6.2. Raychaudhuri's identity

:::{prf:theorem} Expansion of a timelike geodesic congruence
:label: appx-raychaudhuri

Let a smooth Lorentzian manifold have dimension $d+1$, signature $(-,+,\ldots,+)$, and a unit timelike geodesic field $u$. Let $h_{ab}=g_{ab}+u_au_b$ and decompose $B_{ab}=\nabla_bu_a$ as

$$
B_{ab}=\frac\theta d h_{ab}+\sigma_{ab}+\omega_{ab},
\qquad\theta=\nabla_au^a,
$$

where $\sigma$ is symmetric trace-free and $\omega$ antisymmetric. Then

$$
D_u\theta=-\frac{\theta^2}{d}-\sigma_{ab}\sigma^{ab}
+\omega_{ab}\omega^{ab}-R_{ab}u^au^b.
$$
:::

:::{prf:proof}
Commute covariant derivatives in $D_u(\nabla_au^a)$ and use $D_uu=0$. The derivative of the zero acceleration term gives

$$
D_u\theta=-B_{ab}B^{ba}-R_{ab}u^au^b.
$$

Orthogonality to $u$ makes the projector trace $d$, while symmetry and trace-freeness remove the cross terms. Thus
$B_{ab}B^{ba}=\theta^2/d+\sigma_{ab}\sigma^{ab}-\omega_{ab}\omega^{ab}$.
Substitution proves the identity with the stated curvature convention; see also {cite}`wald1984general`. Accelerated congruences have additional acceleration terms.
:::

(sec-appx-geometric-gas-transport)=
### 6.3. Reynolds transport and Voronoi faces

:::{prf:lemma} Reynolds transport for a fixed metric
:label: appx-reynolds-transport

Let $\Omega(t)$ be a smoothly moving domain in a fixed Riemannian manifold, with piecewise smooth boundary normal velocity $w\cdot n$. Then

$$
\frac d{dt}\int_{\Omega(t)}f\,dV_g
=\int_{\Omega(t)}\partial_tf\,dV_g
+\int_{\partial\Omega(t)}f\,w\cdot n\,dA_g.
$$

For a time-dependent metric, add $\tfrac12\int_{\Omega(t)}f\operatorname{tr}_g(\partial_tg)dV_g$.
:::

:::{prf:proof}
Pull the integral back by a flow extending the boundary velocity. Differentiate its Jacobian using $\partial_tJ=J\operatorname{div}_gw$, then apply the divergence theorem. If the metric varies, differentiating $\sqrt{|g|}$ gives $\partial_t\sqrt{|g|}=\tfrac12\sqrt{|g|}\operatorname{tr}_g(\partial_tg)$.
:::

:::{prf:lemma} Voronoi-face normal velocity
:label: appx-voronoi-boundary-velocity

Let $z_i(t),z_j(t)$ lie in a convex normal neighborhood and define

$$
\psi(x,t)=\frac12d_g^2(x,z_i(t))-\frac12d_g^2(x,z_j(t)).
$$

At a regular point of the face $\psi=0$, with $n_{ij}=\nabla\psi/|\nabla\psi|$,

$$
w\cdot n_{ij}=-\frac{\partial_t\psi}{|\nabla\psi|}.
$$

In Euclidean space, writing $r=|z_j-z_i|$, $m=(z_i+z_j)/2$, and $u_i=\dot z_i$,

$$
w\cdot n_{ij}
=\frac{u_i+u_j}{2}\cdot n_{ij}
-\frac{(u_j-u_i)\cdot(x-m)}r.
$$

For a shape-controlled curved configuration of size $\epsilon$, with separation at least $c\epsilon$, bounded curvature $K$, and a $C^1$ velocity field with $\|u\|\leq U_*$ and $\|\nabla u\|\leq L$, the deviation from the averaged transported normal velocity is $O(L\epsilon+KU_*\epsilon^2)$.
:::

:::{prf:proof}
Differentiate $\psi(x(t),t)=0$. In Euclidean space, $\psi=(z_j-z_i)\cdot(x-m)$; differentiating at fixed $x$ gives the exact formula. The correction is bounded by $|u_j-u_i||x-m|/r\leq CL\epsilon$.
In midpoint normal coordinates, first derivatives of squared distance differ from their Euclidean counterparts by $O(K\epsilon^3)$. Divide by $|\nabla\psi|\asymp\epsilon$, retaining the velocity factor $U_*$. This yields the curvature contribution. Vectors in the comparison are parallel transported to the common coordinate frame.
:::

:::{prf:lemma} Divergence remainder on a small cell
:label: appx-divergence-remainder

Let a cell $\Omega$ have diameter at most $C\epsilon$ and volume at most $C\epsilon^d$ in a fixed normal neighborhood. For a $C^2$ vector field $u$ and $x_0\in\Omega$,

$$
\int_{\partial\Omega}u\cdot n\,dA
=\operatorname{Vol}(\Omega)\operatorname{div}u(x_0)+E,
\qquad
|E|\leq C\epsilon^{d+1}\|\nabla\operatorname{div}u\|_{L^\infty(\Omega)}.
$$
:::

:::{prf:proof}
The divergence theorem is exact. Bound
$|\operatorname{div}u(x)-\operatorname{div}u(x_0)|$ by the distance times the supremum of its gradient and integrate. No cell symmetry or vanishing first moment is required.
:::

(sec-appx-geometric-gas-discrete-raychaudhuri)=
### 6.4. Expansion estimates for transported cells

:::{prf:theorem} Discrete Raychaudhuri under differentiated volume consistency
:label: appx-discrete-raychaudhuri

Let $z_i(t)$ follow the geodesic congruence of {prf:ref}`appx-raychaudhuri` and let $V_i(t)>0$ represent transverse cell volumes. Define $\theta_i=\dot V_i/V_i$. Suppose on a fixed time interval

$$
\theta_i(t)=\theta(z_i(t))+r_i(t),
\qquad |r_i(t)|+|\dot r_i(t)|\leq C\epsilon_N,
$$

and the continuum expansion and geometric coefficients are bounded. Then

$$
\dot\theta_i=-\frac{\theta_i^2}{d}-\sigma^2(z_i)+\omega^2(z_i)
-R_{ab}(z_i)u^au^b+O(\epsilon_N).
$$

For material transverse cells where $\dot V_i=\int_{\Omega_i}\theta\,dV$ and material transport applies, the consistency hypothesis follows from diameter $O(\epsilon_N)$ and uniform bounds on the spatial gradients of $\theta$ and $D_u\theta$. A reconstructed Voronoi family additionally needs control of its normal-flux defect and the time derivative of that defect.
:::

:::{prf:proof}
Differentiate the consistency relation and apply the continuous Raychaudhuri identity. Since $\theta_i-\theta(z_i)=O(\epsilon_N)$ and both are bounded, their squared terms differ by $O(\epsilon_N)$, proving the first assertion.

For material cells, $\theta_i$ is the cell average of $\theta$. Its difference from $\theta(z_i)$ is $O(\epsilon_N)$ by the diameter bound. Reynolds transport gives

$$
\dot\theta_i=\langle D_u\theta\rangle_{\Omega_i}
+\langle\theta^2\rangle_{\Omega_i}-\langle\theta\rangle_{\Omega_i}^2.
$$

The first term differs from $D_u\theta(z_i)$ by $O(\epsilon_N)$, and the variance is $O(\epsilon_N^2)$. This proves the differentiated consistency estimate.

For Voronoi cells in a fixed spatial metric, define
$E_i=\int_{\partial\Omega_i}(w-u)\cdot n\,dA$.
Their normalized volume rate has the extra term $E_i/V_i$. Bounds $|E_i/V_i|+|d(E_i/V_i)/dt|=O(\epsilon_N)$, together with the corresponding transport consistency, give the same argument. Shape regularity and the pointwise face estimate alone do not imply these normalized bounds: area scales as $\epsilon_N^{d-1}$ while volume scales as $\epsilon_N^d$. A bound $r_i=O(\epsilon_N)$ alone also cannot be differentiated without derivative control.
:::

:::{div} feynman-prose
A material cell follows the flow; a Voronoi cell is reconstructed from its sites. Their boundaries generally move differently. Reynolds transport tells us exactly where that difference enters: the boundary flux. Once the flux and its time derivative are controlled at the required scale, the continuum expansion identity transfers to the cells.

The holonomy and Raychaudhuri results describe the metric that has been supplied. They do not identify a particular gauge group or derive gravitational field equations. Those identifications require the additional constructions in the fields chapters, including their stated continuum assumptions.
:::

(sec-gg-appendix-b)=
## 7. How to use the estimates together

:::{div} feynman-prose
The proof chain begins with normalized measurements and a positive spectral margin. Those give well-defined geometric coefficients. The backward-generator calculations then give moment control for the specified evolution. Recurrence and a quantitative minorization identify a conservative invariant law; a killed process uses its QSD criterion and survival normalization.

For entropy, use the full-gradient inequality of the actual joint law and the complete modified-entropy derivative for that same law. The recovered kinetic, bounded-density, joint-curvature, and contractive-flow results provide complete analytic routes in their stated regimes. The geometric comparison changes the form constant only after the law has been identified.
:::

:::{admonition} Constants and their roles
:class: feynman-added note

| Estimate | Required input | Consequence |
|---|---|---|
| Velocity ellipticity | $\epsilon_\Sigma-\Lambda_->0$ and a Hessian ceiling | $c_{\min},c_{\max}$ |
| Coefficient derivatives | Full normalized measurement bounds and spectral margin | Matrix derivative bounds |
| Moment stability | Actual backward-generator or kernel drift | Moment envelope and nonexplosion |
| Conservative mixing | Drift and quantitative minorization | Harris convergence |
| Static entropy control | Identified joint-law LSI criterion | Full-gradient LSI |
| Kinetic or conditioned entropy decay | Complete modified-Fisher derivative | Hypocoercive rate |
| Mean-field LSI | Uniform joint LSI and identified marginal limit | Same limiting constant |
| Cell expansion identity | Continuum congruence and differentiated volume consistency | Discrete Raychaudhuri estimate |

A rate is independent of $N$ when every constant entering that rate is uniformly controlled. Velocity ellipticity alone does not make all subsequent constants uniform.
:::
