# Hypocoercive Entropy: From Kinetic Mixing to the Full Swarm

(sec-kl-hypocoercive-overview)=
## 1. What the entropy argument proves

:::{div} feynman-prose
Noise acts directly on velocity. Position moves because velocity carries it. To measure both effects, we combine relative entropy with derivatives in position and velocity, including a cross term that records their coupling. This modified entropy has an explicit exponential decay rate for the conservative kinetic dynamics, including nonconvex confining potentials.

Three quantities determine the estimate: the logarithmic Sobolev constant of the reference law, the curvature bound on the force, and the diffusion and friction coefficients. For independent kinetic particles, tensorization makes the rate independent of population size. The same conclusion holds for interacting laws when their joint inequalities have uniform constants.

Cloning and killing enter through their actual generators. A quasi-stationary distribution is stationary after conditioning on survival; its unnormalized mass decays. The normalization therefore contributes to entropy evolution. We derive that contribution below and state the precise estimate that closes the full swarm argument.

The complete static LSI proofs, nonconvex confinement argument, and extensions are in {doc}`15_kl_convergence`. Finite-particle recurrence, the discrete QSD, and the mean-field equation are developed in {doc}`06_convergence`, {doc}`07_discrete_qsd`, and {doc}`08_mean_field`. Here we work through the kinetic calculation and show how those results can be combined.
:::

:::{admonition} Reading the estimates
:class: feynman-added note

The reference law, its state space, and the evolution must be fixed together. The kinetic Gibbs law, the finite-particle QSD, a stationary mean-field law, and the invariant law of a numerical kernel each have their own defining equation. A proof transfers between them through an identified density, generator, or measure-comparison estimate.
:::

(sec-kl-hypocoercive-reference)=
## 2. Reference law, operators, and modified entropy

:::{div} feynman-prose
Start with the part of the dynamics whose equilibrium density we can calculate. Friction removes velocity fluctuations, noise replenishes them, and Hamiltonian transport preserves total energy. Their balance produces the kinetic Gibbs law. This gives us a reference against which every integration by parts can be checked.

The calculation uses nondimensional position, velocity, and time coordinates. With physical units restored, the coefficients multiplying spatial, velocity, and mixed derivatives carry the corresponding conversion factors. Entropy is measured in nats; a decay rate has units of inverse time.
:::

### 2.1. Conservative kinetic dynamics

:::{prf:definition} Kinetic law and relative-density operator
:label: def-hypocoercive-kinetic-reference

Let $x,v\in\mathbb R^m$, where $m=dN$ for a continuous $N$-particle state. Consider

$$
dX_t=V_tdt,\qquad
dV_t=-\nabla U_N(X_t)dt-\gamma V_tdt+\sqrt{2D}\,dW_t,
\qquad D>0,\quad\gamma>0.
$$

Set $\theta=D/\gamma=\sigma_v^2/(2\gamma)$. When normalizable, its invariant law is

$$
\pi_N(dx,dv)=Z_N^{-1}
\exp\!\left[-\frac{U_N(x)+|v|^2/2}{\theta}\right]dx\,dv.
$$

For $f_t=h_t\pi_N$, the forward equation becomes

$$
\partial_t h=\mathcal Kh,
\qquad
\mathcal K=\mathcal S+\mathcal T,
\qquad
\mathcal S=D\Delta_v-\gamma v\cdot\nabla_v,
\qquad
\mathcal T=-v\cdot\nabla_x+\nabla U_N\cdot\nabla_v.
$$
:::

:::{prf:proof}
The density derivatives are $\nabla_x\log\pi_N=-\nabla U_N/\theta$ and $\nabla_v\log\pi_N=-v/\theta$. Substitution into $\pi_N^{-1}L_{\mathrm{kin}}^*(\pi_Nh)$ gives $\mathcal K$; the zeroth-order terms cancel because $D=\gamma\theta$. Integration by parts yields

$$
\int g\mathcal Sh\,d\pi_N=-D\int\nabla_vg\cdot\nabla_vh\,d\pi_N,
\qquad
\int g\mathcal Th\,d\pi_N=-\int h\mathcal Tg\,d\pi_N.
$$

The second identity follows because $\operatorname{div}(\pi_N(-v,\nabla U_N))=0$. These identities also verify invariance. They hold first for smooth functions with justified integration by parts, then on the corresponding operator domains.
:::

### 2.2. Entropy and the three Fisher terms

:::{prf:definition} Modified entropy
:label: def-hypocoercive-entropy-functional

For $h>0$ with $\pi_N(h)=1$, put $u=\log h$ and define

$$
\begin{aligned}
H(h)&=\int h\log h\,d\pi_N,\\
I_x(h)&=\int h|\nabla_xu|^2d\pi_N,\qquad
I_v(h)=\int h|\nabla_vu|^2d\pi_N,\\
I_{xv}(h)&=\int h\nabla_xu\cdot\nabla_vu\,d\pi_N,
\qquad I(h)=I_x(h)+I_v(h).
\end{aligned}
$$

Let

$$
G=\begin{pmatrix}cI&bI\\bI&aI\end{pmatrix},
\qquad a,c>0,\quad ac>b^2,
\qquad
\Phi_G(h)=H(h)+cI_x(h)+2bI_{xv}(h)+aI_v(h).
$$

Writing $q=(\nabla_xu,\nabla_vu)^{\mathsf T}$, the gradient contribution is $I_G(h)=\int hq^{\mathsf T}Gq\,d\pi_N$. If $G\preceq g_+I$ and

$$
\operatorname{Ent}_{\pi_N}(f^2)\leq2C_N\int|\nabla f|^2d\pi_N,
$$

then

$$
H(h)\leq\Phi_G(h)\leq(C_N/2+g_+)I(h).
$$
:::

:::{prf:proof}
Positive definiteness of $G$ makes $I_G$ nonnegative. Apply the LSI to $f=\sqrt h$ to obtain $H(h)\leq C_NI(h)/2$, and use $I_G\leq g_+I$.
:::

:::{div} feynman-prose
The cross term may be negative. What must stay positive is the entire quadratic form, which is exactly what $ac>b^2$ ensures. Its derivative will produce the spatial dissipation that ordinary entropy misses.

The LSI uses all position and velocity derivatives. The kinetic generator itself has only velocity diffusion. A nonconstant test function depending solely on position has zero velocity gradient and positive entropy, so a velocity-only LSI cannot hold for the full phase-space law. The coupling calculation is what connects velocity noise to spatial information.
:::

### 2.3. Obtaining the LSI without global convexity

:::{prf:corollary} Explicit kinetic reference LSI
:label: thm-unconditional-lsi-explicit

Suppose the spatial Gibbs law $\pi_x\propto e^{-U/\theta}$ satisfies an LSI with constant $C_x$. Then $m_U=\pi_x\otimes\mathcal N(0,\theta I_d)$ satisfies a full-gradient LSI with

$$
C_0=\max\{C_x,\theta\}.
$$

The same constant holds for $m_U^{\otimes N}$. Two sufficient nonconvex regimes are:

1. $U=W+B$, with $\nabla^2W\succeq\kappa I$, $\kappa>0$, and bounded $\operatorname{osc}(B)$. One may take $C_x=\theta e^{\operatorname{osc}(B)/\theta}/\kappa$.
2. $U\in C^2$, $x\cdot\nabla U(x)\geq\alpha_U|x|^2-b_U$ with $\alpha_U>0$, and $\nabla^2U\succeq-M_-I$. The explicit finite constant constructed in {prf:ref}`thm-unconditional-lsi` applies.
:::

:::{prf:proof}
In the first regime, Bakry–Émery gives constant $\theta/\kappa$ for $e^{-W/\theta}$; bounded density perturbation multiplies it by at most $e^{\operatorname{osc}(B)/\theta}$. In the second, the Lyapunov function

$$
\mathcal W(x)=\exp\!\left(\frac{\alpha_U|x|^2}{4\theta}\right)
$$

satisfies a negative quadratic drift for $\Delta-\theta^{-1}\nabla U\cdot\nabla$. The weighted-energy estimate, local Poincaré inequality, entropy-transport bound, and centering argument in {prf:ref}`thm-nonconvex-main` prove the spatial LSI. Gaussian LSI and the full tensorization proof in {prf:ref}`thm-tensorization` then give $C_0$, independently of $N$.
:::

:::{prf:remark} The actual joint law
:label: rem-hypocoercive-joint-lsi

For interacting laws, {prf:ref}`cor-n-uniform-lsi` supplies four proved criteria: a product reference, a uniformly bounded joint density tilt, uniform joint curvature, or a uniformly contractive additive-noise invariant flow. A QSD can use this corollary when its actual law satisfies one of these criteria. Conditional laws on continuous status strata also require the discrete entropy accounting in {prf:ref}`prop-kl-status-entropy` when the strata are combined.
:::

(sec-kl-hypocoercive-kinetic-proof)=
## 3. Complete kinetic dissipation calculation

:::{div} feynman-prose
Differentiating velocity information creates a position–velocity cross term. Differentiating that cross term creates negative spatial information. This is the transfer mechanism. The second derivatives generated by diffusion must be collected into one quadratic form; the mixed second-derivative term has no sign by itself.

We use a global force-Hessian bound in this calculation. A compact region containing most of the probability does not supply that bound for a Fisher integral, which weights squared density gradients. If only local curvature bounds are available, the tail terms need a weighted estimate of their own.
:::

### 3.1. Exact identities

:::{prf:lemma} Evolution of entropy and Fisher information
:label: lem-hypocoercive-exact-identities

Suppose $\|\nabla^2U_N\|_{\mathrm{op}}\leq M$ globally. For a smooth positive relative density evolving under $\mathcal K$, with finite integrals and justified integration by parts, write $H_U=\nabla^2U_N$. Then

$$
\begin{aligned}
\dot H&=-DI_v,\\
\dot I_v&=-2\gamma I_v-2I_{xv}
-2D\int h\|\nabla_v^2u\|_{\mathrm{HS}}^2d\pi_N,\\
\dot I_{xv}&=-I_x-\gamma I_{xv}
+\int h(\nabla_vu)^{\mathsf T}H_U\nabla_vu\,d\pi_N\\
&\quad-2D\int h\sum_j\partial_{v_j}\nabla_xu\cdot\partial_{v_j}\nabla_vu\,d\pi_N,\\
\dot I_x&=2\int h(\nabla_xu)^{\mathsf T}H_U\nabla_vu\,d\pi_N
-2D\int h\|\nabla_v\nabla_xu\|_{\mathrm{HS}}^2d\pi_N.
\end{aligned}
$$
:::

:::{prf:proof}
Symmetry of $\mathcal S$, antisymmetry of $\mathcal T$, and the diffusion chain rule give $\dot H=-DI_v$. The derivative commutators are

$$
[\nabla_v,\mathcal K]=-\nabla_x-\gamma\nabla_v,
\qquad [\nabla_x,\mathcal K]=H_U\nabla_v.
$$

Set $B=\begin{pmatrix}0&H_U\\-I&-\gamma I\end{pmatrix}$. Since $\partial_tu=\mathcal Ku+D|\nabla_vu|^2$ and $\nabla\mathcal Ku=\mathcal Kq+Bq$, differentiating $I_G$ gives

$$
\frac{d}{dt}I_G
=-2D\int h\sum_j(\partial_{v_j}q)^{\mathsf T}G(\partial_{v_j}q)d\pi_N
+\int hq^{\mathsf T}(GB+B^{\mathsf T}G)q\,d\pi_N.
$$

Indeed, integration by parts combines the terms containing $\mathcal K$ into the displayed second-derivative form. The derivative of $D|\nabla_vu|^2$ cancels its accompanying first-gradient terms. Reading off the velocity, cross, and position entries gives the three identities; this is the calculation of {prf:ref}`lem-kinetic-evolution-bounds` with the same conventions.
:::

### 3.2. Positive coefficients and a rigorous rate

:::{prf:theorem} Explicit hypocoercive decay rate
:label: thm-explicit-kinetic-decay

Suppose the kinetic Gibbs law $\pi_N$ satisfies a full-gradient LSI with constant $C_N$, and $\|\nabla^2U_N\|_{\mathrm{op}}\leq M$. Define

$$
L_M=2M+\gamma+2,
\qquad \eta=\frac{D}{2(1+2M+L_M^2)},
\qquad G=\eta\begin{pmatrix}2I&I\\I&2I\end{pmatrix}.
$$

Then

$$
\frac{d}{dt}\Phi_G(h_t)
\leq-\eta I_x(h_t)-\frac D2I_v(h_t)
\leq-\eta I(h_t),
$$

and

$$
\Phi_G(h_t)\leq e^{-r_Nt}\Phi_G(h_0),
\qquad H(h_t)\leq e^{-r_Nt}\Phi_G(h_0),
\qquad r_N=\frac{\eta}{C_N/2+3\eta}>0.
$$

The estimate holds for finite-$\Phi_G$ initial data, and from any positive time $t_0$ at which $\Phi_G(h_{t_0})<\infty$.
:::

:::{prf:proof}
The matrix has eigenvalues $\eta$ and $3\eta$. Thus the second-derivative term in the preceding lemma is nonpositive. The Hessian bound and Cauchy–Schwarz give

$$
\left|I_{xv}\right|\leq\sqrt{I_xI_v},\quad
\int h(\nabla_vu)^{\mathsf T}H_U\nabla_vu\,d\pi_N\leq MI_v,
$$

$$
\left|\int h(\nabla_xu)^{\mathsf T}H_U\nabla_vu\,d\pi_N\right|
\leq M\sqrt{I_xI_v}.
$$

With $a=c=2\eta$ and $b=\eta$, the identities therefore imply

$$
\dot\Phi_G\leq-2\eta I_x-(D+4\eta\gamma-2\eta M)I_v
+2\eta L_M\sqrt{I_xI_v}.
$$

Apply $2\eta L_M\sqrt{I_xI_v}\leq\eta I_x+\eta L_M^2I_v$:

$$
\dot\Phi_G\leq-\eta I_x-
[D+4\eta\gamma-\eta(2M+L_M^2)]I_v.
$$

The chosen $\eta$ satisfies $\eta(2M+L_M^2)\leq D/2$ and $\eta\leq D/2$, which proves dissipation. The LSI gives $\Phi_G\leq(C_N/2+3\eta)I$. Grönwall's inequality proves the decay estimate. Approximation extends the calculation from smooth data to its finite-functional domain, as in {prf:ref}`thm-villani-hypocoercivity`.
:::

:::{div} feynman-prose
There are no approximate signs in this rate. It is a sufficient lower bound, not an optimized rate. The initial factor is the modified entropy: a small coefficient in front of Fisher information does not make arbitrary gradients small.

Nonconvexity enters through two different quantities. The Hessian norm controls the dynamical mixing calculation; the LSI constant also records confinement and barriers between wells. The preceding spatial proof supplies that constant without requiring global convexity.
:::

(sec-kl-hypocoercive-full-evolution)=
## 4. Cloning and conditioning on survival

:::{div} feynman-prose
A jump changes an entire swarm state. Its transition kernel must include the sampled companions, acceptance rule, and correlated offspring perturbations. The finite-particle kernel is a linear Markov operator on swarm laws even when each transition uses empirical fitness. A closed one-particle equation requires a separate mean-field argument.

For conservative dynamics, an invariant reference makes entropy decrease under a Markov step by data processing. For killed dynamics, we must also divide by the surviving mass. That division changes the derivative, even when the conditional law has already reached its QSD.
:::

### 4.1. A common invariant target

:::{prf:theorem} Kinetic dynamics with a common-target cloning kernel
:label: thm-hypocoercive-common-target-cloning

Let $P$ be a Markov kernel preserving the kinetic reference $\pi_N$, and let $P^\dagger$ be its action on relative densities. Suppose

$$
I_G(P^\dagger h)\leq A_JI_G(h)
$$

for every density in the form domain. Add jumps at rate $\omega\geq0$, so $\partial_th=\mathcal Kh+\omega(P^\dagger h-h)$. If

$$
\delta=\eta-3\eta\omega(A_J-1)_+>0,
$$

then

$$
H(h_t)\leq\Phi_G(h_t)
\leq\exp\!\left[-\frac{\delta t}{C_N/2+3\eta}\right]\Phi_G(h_0).
$$
:::

:::{prf:proof}
Data processing gives $H(P^\dagger h)\leq H(h)$ because $\pi_NP=\pi_N$. Entropy and $I_G=\int(\nabla h)^{\mathsf T}G\nabla h/h\,d\pi_N$ are convex in $h$, the latter by the convexity of a quadratic perspective. Hence the directional derivative along the jump generator is bounded by

$$
D\Phi_G(h)[\omega(P^\dagger h-h)]
\leq\omega(A_J-1)_+I_G(h)
\leq3\eta\omega(A_J-1)_+I(h).
$$

Combine this with the kinetic dissipation and the LSI closure. The full kernel and domain requirements are those of {prf:ref}`thm-main-kl-convergence`.
:::

### 4.2. The normalized QSD entropy identity

:::{prf:proposition} Entropy under killed diffusion and jumps
:label: prop-hypocoercive-qsd-entropy

Let the conservative backward generator on a continuous swarm state space be

$$
LF(S)=b(S)\cdot\nabla F(S)+\operatorname{tr}(a(S)\nabla^2F(S))
+\int[F(S')-F(S)]r(S,dS'),
$$

with $a\succeq0$. Let $A=L-\kappa$ for a bounded nonnegative killing rate, on a domain without absorbing boundary flux. Let $\nu_N$ be a QSD satisfying

$$
L^*\nu_N=(\kappa-\lambda_N)\nu_N,
\qquad\lambda_N=\nu_N(\kappa).
$$

For $f_t=h_t\nu_N$, the conditioned equation is

$$
\partial_tf_t=L^*f_t-\kappa f_t+\bar\kappa_t f_t,
\qquad\bar\kappa_t=f_t(\kappa).
$$

Set $\psi(h)=h\log h-h+1$ and $\mathcal B(s,t)=s\log(s/t)-s+t$. Then

$$
\begin{aligned}
\mathcal D_{\nu_N}(h)
&=\int\frac{(\nabla h)^{\mathsf T}a\nabla h}{h}\,d\nu_N\\
&\quad+\iint\mathcal B(h(S),h(S'))\,\nu_N(dS)r(S,dS'),
\end{aligned}
$$

and the exact entropy identity is

$$
\frac{d}{dt}H_{\nu_N}(h_t)
=-\mathcal D_{\nu_N}(h_t)
+\bar\kappa_tH_{\nu_N}(h_t)-\nu_N[\kappa\psi(h_t)].
$$

In particular, $\dot H\leq-\mathcal D_{\nu_N}+\operatorname{osc}(\kappa)H$.
:::

:::{prf:proof}
Normalize the unnormalized solution $\partial_t\widetilde f=A^*\widetilde f$ by its mass to obtain the displayed equation. The diffusion and jump chain rules give

$$
hL\log h=Lh-\frac{(\nabla h)^{\mathsf T}a\nabla h}{h}
-\int\mathcal B(h(S),h(S'))r(S,dS').
$$

Integrate against $\nu_N$, use its eigenmeasure equation, and differentiate $H_{\nu_N}(h)=\int h\log h\,d\nu_N$. The killing terms combine into $\bar\kappa_tH-\nu_N(\kappa\psi(h))$. Since $\psi\geq0$, $\nu_N\psi(h)=H$, and $\bar\kappa_t\leq\sup\kappa$, the upper bound follows. This is {prf:ref}`prop-kl-conditioned-entropy`.
:::

:::{prf:remark} Absorbing boundary flux
:label: rem-hypocoercive-boundary-flux

For a kinetic absorbing boundary with outgoing set $\Gamma_+=\{(x,v):v\cdot n(x)>0\}$, the total mass-loss rate is

$$
\ell_t=f_t(\kappa)+\int_{\Gamma_+}(v\cdot n)f_t.
$$

Under the trace and domain hypotheses of {prf:ref}`prop-kl-boundary-entropy`, replace $\bar\kappa_t$ by $\ell_t$ and add
$-\int_{\Gamma_+}(v\cdot n)\nu_N\psi(h_t)$ to the entropy identity. A confinement moment estimate does not remove these boundary terms.
:::

### 4.3. Closure for the full conditioned generator

:::{prf:corollary} Actual-law hypocoercive convergence
:label: cor-hypocoercive-full-qsd

Let the actual continuous-law QSD $\nu_N$ satisfy a full-gradient LSI with constant $C_N$. Suppose $G_N\succ0$, $G_N\preceq g_+I$, and the complete normalized evolution, including cloning and any killing or boundary contributions, satisfies

$$
\frac{d}{dt}\Phi_{G_N}(h_t)\leq-\delta_NI_{\nu_N}(h_t),
\qquad\delta_N>0.
$$

Then

$$
H_{\nu_N}(h_t)\leq\Phi_{G_N}(h_t)
\leq e^{-\delta_Nt/(C_N/2+g_+)}\Phi_{G_N}(h_0).
$$
:::

:::{prf:proof}
The LSI gives $\Phi_{G_N}\leq(C_N/2+g_+)I_{\nu_N}$. Substitute into the full derivative bound and apply Grönwall. The first-variation formula needed to check that derivative is {prf:ref}`lem-kl-functional-first-variation`; the complete theorem is {prf:ref}`thm-kl-convergence-euclidean`.
:::

:::{prf:remark} Alternative through the conditioned process
:label: rem-hypocoercive-doob-route

A positive right survival eigenfunction yields a conservative Doob-transformed process with invariant law proportional to that eigenfunction times the QSD. The exact conjugacy and entropy transfer are proved in {prf:ref}`prop-kl-doob-transform` and {prf:ref}`thm-main-kl-final`. Uniform transfer requires the stated survival-eigenfunction ratio bounds, or a replacement weighted comparison. This provides another route to the actual conditional law when those estimates hold.
:::

(sec-kl-hypocoercive-selection-forcing)=
## 5. Selection gradients and persistent forcing

:::{div} feynman-prose
A selection equation can make a density sharper. A derivative bound must distinguish growth proportional to existing gradients from a source that creates gradients even at the reference density. The latter produces a square-root term in Fisher information. Keeping that term determines whether the calculation proves convergence to the reference or a bound around it.
:::

:::{prf:proposition} Exact derivative for a normalized multiplication model
:label: prop-hypocoercive-selection-derivative

For this proposition only, consider the specified selection equation

$$
\partial_tf=\omega(V/\bar V_f-1)f,
\qquad\bar V_f=\int V f,
$$

with a fixed smooth $V\geq v_*>0$, and a fixed positive reference $\pi$. Let $h=f/\pi$, $c_f=\omega(V/\bar V_f-1)$, and $G$ be constant positive definite. Then

$$
\dot H_\pi(h)=\frac{\omega}{\bar V_f}
\operatorname{Cov}_f(V,\log h),
$$

$$
\dot I_G(h)=\int c_f hq^{\mathsf T}Gq\,d\pi
+\frac{2\omega}{\bar V_f}\int hq^{\mathsf T}G\nabla V\,d\pi,
\qquad q=\nabla\log h.
$$

If $|V/\bar V_f-1|\leq K$ and $(\nabla V)^{\mathsf T}G\nabla V\leq S_G^2$, then

$$
\dot I_G\leq\omega K I_G+\frac{2\omega S_G}{v_*}\sqrt{I_G}.
$$
:::

:::{prf:proof}
Mass preservation gives $\int c_f f=0$, so differentiating entropy gives the covariance formula. Since $\dot h=c_fh$ and $\partial_t\nabla\log h=\nabla c_f=\omega\nabla V/\bar V_f$,

$$
\frac{d}{dt}\int hq^{\mathsf T}Gq\,d\pi
=\int c_fhq^{\mathsf T}Gq\,d\pi
+2\int hq^{\mathsf T}G\nabla c_f\,d\pi.
$$

Weighted Cauchy–Schwarz and the stated bounds prove the estimate.
:::

:::{prf:remark} Scope of the multiplication model
:label: rem-hypocoercive-selection-model

This equation is a specified normalized selection model. Identifying it with an algorithm's mean-field limit requires deriving it from the actual sampled companion and acceptance law; see {doc}`08_mean_field`. Its entropy covariance has no general negative sign. Moreover, $f=\pi$ is stationary for this selection term only when $V$ is constant $\pi$-almost everywhere. Kinetic and selection terms may balance at a full stationary law, but their separate reference-law cancellations must then be recomputed.
:::

:::{prf:lemma} A square-root forcing term leaves a quantitative floor
:label: lem-hypocoercive-forcing-floor

Suppose a nonnegative functional satisfies $\Phi\leq C_I I$ and

$$
\dot\Phi\leq-aI+B\sqrt I+E,
\qquad a,C_I>0,\quad B,E\geq0.
$$

Then, with $r=a/(2C_I)$,

$$
\Phi(t)\leq e^{-rt}\Phi(0)
+\frac{B^2/(2a)+E}{r}(1-e^{-rt}).
$$
:::

:::{prf:proof}
Young's inequality gives $B\sqrt I\leq(a/2)I+B^2/(2a)$. Thus $\dot\Phi\leq-r\Phi+B^2/(2a)+E$. Integrate this scalar differential inequality.
:::

:::{div} feynman-prose
For the common-invariant-kernel theorem, the jump estimate is proportional to existing information, so a strict margin gives decay to zero. A persistent source instead leaves the displayed floor. An estimate for spatial Fisher alone also leaves the entropy and cross-Fisher derivatives to be controlled. These distinctions determine the actual parameter condition; a friction-versus-cloning slogan cannot replace them.
:::

(sec-kl-hypocoercive-discrete-population)=
## 6. Discrete time and population-independent constants

:::{div} feynman-prose
A numerical step is a new transition operator. To transfer the continuous estimate, bound its error in the same modified entropy. An error bound for smooth observables does not by itself control density gradients. Once the functional error is known, the iteration is elementary and its accumulated floor is explicit.

Population size enters through the constants in the joint-law estimates. Tensorization proves those constants for a product reference while allowing correlated initial densities. For interacting swarms, the corresponding joint estimates must be uniform. Propagation of chaos identifies marginal limits; it is not a finite-particle spectral-gap comparison.
:::

### 6.1. A discrete functional estimate

:::{prf:lemma} Discrete entropy decay with the numerical defect retained
:label: lem-discrete-entropy-decay

Let $\mathcal T_\tau$ be a continuous evolution satisfying $\Phi(\mathcal T_\tau h)\leq e^{-r\tau}\Phi(h)$. Let $P_\tau$ denote an actual numerical update, expressed relative to the same reference law. Suppose its functional defect satisfies

$$
\Phi(P_\tau h)-\Phi(\mathcal T_\tau h)
\leq K\tau^{p+1}\Phi(h)+B\tau^{p+1},
\qquad p>0.
$$

If $q_\tau=e^{-r\tau}+K\tau^{p+1}<1$, then

$$
\Phi(h_n)\leq q_\tau^n\Phi(h_0)
+\frac{B\tau^{p+1}}{1-q_\tau}(1-q_\tau^n).
$$

For constants uniform as $\tau\downarrow0$, the floor is $O(\tau^p)$. If $B=0$, the estimate gives convergence to the chosen reference.
:::

:::{prf:proof}
The two assumed inequalities give $\Phi(h_{n+1})\leq q_\tau\Phi(h_n)+B\tau^{p+1}$. Iterate the scalar recurrence and sum the geometric series. Since $1-q_\tau=r\tau+o(\tau)$, the floor has the stated order. This is {prf:ref}`lem-discrete-lsi-from-curvature`.
:::

:::{prf:remark} Splitting order and the numerical target
:label: rem-hypocoercive-numerical-target

An $O(\tau^3)$ local defect yields an $O(\tau^2)$ functional floor when the lemma's hypotheses hold. A Lie–Trotter local operator defect is generally $O(\tau^2)$; integrating against a density functional does not automatically improve that order. BAOAB and cloning composition need estimates for their actual operators and domains. If the numerical invariant law or QSD differs from the continuous reference, identifying its entropy distance requires the corresponding measure comparison. The discrete QSD construction is in {doc}`07_discrete_qsd`.
:::

### 6.2. Uniformity in the number of particles

:::{prf:theorem} Population-independent hypocoercive rates
:label: thm-n-uniformity

For product kinetic potentials $U_N=\sum_{i=1}^N U(x_i)$, suppose the one-particle hypotheses of {prf:ref}`thm-unconditional-lsi-explicit` hold and $\sup_x\|\nabla^2U(x)\|_{\mathrm{op}}\leq M$. With fixed $D,\gamma$, the rate in {prf:ref}`thm-explicit-kinetic-decay` is

$$
r_* = \frac{\eta}{C_0/2+3\eta}>0,
$$

independently of $N$, for arbitrary initial joint densities with finite modified entropy.

More generally, for actual interacting invariant laws or QSDs satisfying {prf:ref}`cor-hypocoercive-full-qsd`, if

$$
\sup_N C_N\leq C_*,\qquad
\sup_N\|G_N\|_{\mathrm{op}}\leq g_*,\qquad
\inf_N\delta_N\geq\delta_*>0,
$$

then their rates are at least $\delta_* /(C_*/2+g_*)$.
:::

:::{prf:proof}
The product reference has LSI constant $C_0$ by tensorization. The Hessian of $U_N$ is block diagonal, so its operator norm is at most $M$. The kinetic proof uses these constants and sums of squared derivatives; neither step introduces a factor of $N$. For the interacting statement, substitute the three uniform bounds into the full-law rate.
:::

:::{prf:corollary} Empirical variance from the joint LSI
:label: cor-hypocoercive-empirical-variance

If an actual joint law $\nu_N$ has the full-gradient LSI constant $C_*$, and $\|\nabla\varphi\|_\infty\leq L$, then

$$
\operatorname{Var}_{\nu_N}\!\left(\frac1N\sum_i\varphi(z_i)\right)
\leq\frac{C_*L^2}{N}.
$$
:::

:::{prf:proof}
Linearizing LSI gives Poincaré with constant $C_*$. Each particle gradient of the empirical average is $N^{-1}\nabla\varphi(z_i)$, so the sum of squared gradients is at most $L^2/N$. This is {prf:ref}`cor-quantitative-lsi-final`; independence is unnecessary once the joint inequality is established.
:::

:::{div} feynman-prose
The final rate has a concrete interpretation: the numerator is the dissipation remaining after the full generator has been estimated, and the denominator converts Fisher information into modified entropy. Increasing friction affects both the mixing coefficient and the kinetic temperature, so the formula does not justify an unlimited increase in friction. Noise likewise changes both dissipation and the reference law.

For a mean-field conclusion, combine the finite-particle estimate with an identified marginal limit and the existence and uniqueness results for its equation. The LSI passes to fixed marginal limits under the conditions of {prf:ref}`cor-kl-lsi-mean-field-limit`. This keeps the finite-particle entropy theorem and the mean-field identification in one chain, with each estimate doing its stated job.
:::
