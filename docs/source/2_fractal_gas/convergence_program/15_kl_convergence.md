(sec-fg-kl-convergence)=
# Logarithmic Sobolev inequalities and entropy convergence

(sec-fg-kl-conv-overview)=
## Overview

:::{div} feynman-prose
An entropy estimate answers a distributional question: how much information about the initial law survives after the swarm evolves? A variance estimate answers a different question: how widely spread are the walkers? A cloud can have the right variance and the wrong distribution. To pass from one question to the other, we need a functional inequality for a specified probability law.

This chapter develops that argument in three stages. First we prove logarithmic Sobolev inequalities, including bounds independent of the number of particles. Next we show how velocity diffusion and position transport combine to dissipate a modified entropy. Finally we incorporate the cloning operator and the normalization required when conditioning on survival. Each stage keeps its reference measure explicit.

The proofs apply on unbounded phase space with a confining potential. Nonconvex potentials are included through quadratic radial confinement and a lower Hessian bound, or through a bounded perturbation of a convex confining potential. The constants retain the dependence on the potential and the interaction strength; increasing the population does not remove that dependence.
:::

:::{admonition} Reading guide
:class: feynman-added note

The main static estimate is {prf:ref}`cor-n-uniform-lsi`. The kinetic argument is proved in {prf:ref}`lem-hypocoercive-dissipation` and {prf:ref}`thm-villani-hypocoercivity`. The exact entropy balance for a killed process is {prf:ref}`prop-kl-conditioned-entropy`; the full evolution is assembled in {prf:ref}`thm-kl-convergence-euclidean`.

The algorithm and its finite-step kernel are specified in {doc}`02_euclidean_gas` and {doc}`../1_the_algorithm/02_fractal_gas_latent`. The complementary drift, mean-field, and regularity estimates are developed in {doc}`06_convergence`, {doc}`08_mean_field`, {doc}`09_propagation_chaos`, and {doc}`17_geometric_gas`.
:::

(sec-fg-kl-conv-laws)=
## 1. Probability laws and entropy

:::{prf:definition} Finite-particle laws and survival conditioning
:label: def-kl-finite-particle-laws

Fix a population size $N$. Let $E_N$ be the nonabsorbed swarm state space. A conservative Markov semigroup $P_t$ has an invariant probability law $\pi_N$ when

$$
\pi_NP_t=\pi_N.
$$

For a killed process, write

$$
Q_tF(S)=\mathbb E_S[F(S_t)\mathbf 1_{\{t<T_\partial\}}].
$$

A quasi-stationary distribution $\nu_N$ satisfies

$$
\nu_NQ_t=e^{-\lambda_Nt}\nu_N,
\qquad
\lambda_N\geq0.
$$

The law conditioned on survival is

$$
\mu_t=\frac{\mu_0Q_t}{\mu_0Q_t1},
\qquad \mu_0Q_t1>0.
$$

In discrete time, replace $Q_t$ by a sub-Markov kernel $Q^n$ and $e^{-\lambda_Nt}$ by $\vartheta_N^n$, where $\nu_NQ=\vartheta_N\nu_N$ and $0<\vartheta_N\leq1$.

A one-particle marginal of $\nu_N$, a limiting mean-field stationary law, and an invariant law of a process that revives killed walkers are separate probability measures. Their identification requires an equation or a limiting theorem.
:::

:::{prf:definition} Relative entropy and full Fisher information
:label: def-relative-entropy

For probability measures $\mu\ll\pi$ on a continuous phase space, put $h=d\mu/d\pi$ and

$$
H_\pi(h)=D_{\mathrm{KL}}(\mu\Vert\pi)
=\int h\log h\,d\pi.
$$

For an unnormalized nonnegative function $g$, define

$$
\operatorname{Ent}_\pi(g)
=\int g\log\!\left(\frac{g}{\pi(g)}\right)d\pi.
$$

For $S=(x_1,v_1,\ldots,x_N,v_N)$, the full Fisher information is

$$
I_\pi(h)=\int h\sum_{i=1}^N
\left(|\nabla_{x_i}\log h|^2+|\nabla_{v_i}\log h|^2\right)d\pi
=I_x(h)+I_v(h).
$$

The integral is understood by lower semicontinuous extension when necessary. Relative entropy is $+\infty$ if absolute continuity fails.
:::

:::{prf:definition} Full-gradient logarithmic Sobolev inequality
:label: def-lsi-continuous

A probability law $\pi$ satisfies an LSI with constant $C_{\mathrm{LSI}}$ if

$$
\operatorname{Ent}_\pi(f^2)
\leq 2C_{\mathrm{LSI}}\int |\nabla f|^2\,d\pi
\tag{15.1}
$$

for every function in the associated Sobolev space. Equivalently, for probability densities $h$,

$$
H_\pi(h)\leq\frac{C_{\mathrm{LSI}}}{2}I_\pi(h).
\tag{15.2}
$$

For $N$ particles, the gradient in (15.1) is the sum of the position and velocity gradients over all particles. An $N$-uniform LSI means that the same upper bound for $C_{\mathrm{LSI}}$ works for all $N$ in the stated family of laws.
:::

:::{prf:remark} Velocity diffusion and the gradient in an LSI
:label: rem-note-kinetic-non-reversibility

For the kinetic generator

$$
L_{\mathrm{kin}}
=v\cdot\nabla_x-\nabla U\cdot\nabla_v
-\gamma v\cdot\nabla_v+\frac{\sigma_v^2}{2}\Delta_v,
$$

the actual carré du champ is

$$
\Gamma_L(f,f)=\frac{\sigma_v^2}{2}|\nabla_v f|^2.
$$

An LSI with only this gradient cannot hold for every phase-space function under a law with a nontrivial spatial marginal: take $f=f(x)$ nonconstant. Its entropy can be positive while $\Gamma_L(f,f)=0$.

The full-gradient inequality (15.1) is a property of the measure. Nonreversibility does not prevent that inequality. It changes how the dynamics use it, which is the purpose of the hypocoercive calculation below.
:::

:::{div} feynman-prose
Imagine perturbing only the spatial density while leaving the conditional velocities unchanged. At the first instant, velocity noise sees no error. Transport then turns spatial variation into position-velocity correlations, and diffusion can act. This delay explains why the useful quantity contains gradients and a cross term, rather than entropy alone.
:::

:::{prf:proposition} Entropy decomposition across status strata
:label: prop-kl-status-entropy

Suppose a joint law has the disintegration $\pi(ds,dz)=p_s\pi_s(dz)$ over alive/dead status patterns $s$. For $g_s=\bigl(\pi_s(f^2)\bigr)^{1/2}$,

$$
\operatorname{Ent}_\pi(f^2)
=\sum_s p_s\operatorname{Ent}_{\pi_s}(f^2)
+\operatorname{Ent}_p(g^2).
\tag{15.3}
$$

Consequently, full-gradient LSIs on the continuous strata control the first term. Controlling the entropy of the entire status-bearing law also requires a functional inequality for the second term, with a discrete form connecting the status patterns.
:::

:::{prf:proof}
Insert $\log(f^2/\pi(f^2))=\log(f^2/\pi_s(f^2))+\log(\pi_s(f^2)/\pi(f^2))$ into the entropy integral and sum over $s$. A function constant on each stratum has zero continuous gradient, so the second term cannot be omitted when more than one status has positive probability.
:::

(sec-fg-kl-conv-static-lsi)=
## 2. Complete proofs of the static LSI

### 2.1. Curvature

:::{prf:theorem} Bakry-Émery criterion
:label: thm-bakry-emery

Let $\pi(dz)=Z^{-1}e^{-V(z)}dz$ on $\mathbb R^m$, with $V\in C^2$ and

$$
\nabla^2V(z)\succeq\rho I_m,
\qquad \rho>0.
$$

Then (15.1) holds with $C_{\mathrm{LSI}}\leq\rho^{-1}$.
:::

:::{prf:proof}
Use the auxiliary elliptic generator $\mathscr L=\Delta-\nabla V\cdot\nabla$, which preserves $\pi$. For a smooth positive probability density $h$, let $h_t=e^{t\mathscr L}h$ and $u_t=\log h_t$. Integration by parts gives

$$
\frac{d}{dt}H_\pi(h_t)=-I_\pi(h_t).
\tag{15.4}
$$

Differentiating the Fisher information, using $\partial_tu_t=\mathscr Lu_t+|\nabla u_t|^2$, gives the exact identity

$$
\frac{d}{dt}I_\pi(h_t)
=-2\int h_t\left(
\|\nabla^2u_t\|_{\mathrm{HS}}^2
+\nabla u_t^{\mathsf T}\nabla^2V\nabla u_t
\right)d\pi.
\tag{15.5}
$$

For completeness, the local identity underlying (15.5) is

$$
\frac12\mathscr L|\nabla u|^2
-\nabla u\cdot\nabla\mathscr Lu
=\|\nabla^2u\|_{\mathrm{HS}}^2
+\nabla u^{\mathsf T}\nabla^2V\nabla u.
$$

Apply the product rule to $\int h_t|\nabla u_t|^2d\pi$; the terms containing $\nabla|\nabla u_t|^2$ cancel after integration by parts, leaving (15.5). Thus $I_\pi(h_t)\leq e^{-2\rho t}I_\pi(h)$.

For bounded $h$ bounded away from zero, ergodicity of this uniformly confining elliptic diffusion and bounded convergence give $H_\pi(h_t)\to0$. One can obtain this ergodicity directly from synchronous coupling: strong convexity implies contraction of two trajectories by $e^{-\rho t}$. Integrating (15.4) therefore yields

$$
H_\pi(h)=\int_0^\infty I_\pi(h_t)\,dt
\leq\frac{I_\pi(h)}{2\rho}.
$$

Apply this to $h=f^2/\pi(f^2)$. Truncation, positive regularization, and lower semicontinuity extend the result to the Sobolev domain. Smooth approximation of $V$ gives the stated $C^2$ formulation.
:::

### 2.2. Tensorization

:::{prf:theorem} Tensorization of the logarithmic Sobolev inequality
:label: thm-tensorization

If $\pi_i$ satisfies (15.1) with constant $C_i$, then $\pi=\bigotimes_{i=1}^N\pi_i$ satisfies it with

$$
C_{\mathrm{LSI}}(\pi)\leq\max_i C_i.
\tag{15.6}
$$

The test function $f$ may depend jointly on every coordinate; it need not factorize.
:::

:::{prf:proof}
For two factors $\pi=\mu\otimes\nu$, set $g(x)=\bigl(\int f(x,y)^2\nu(dy)\bigr)^{1/2}$. The entropy chain rule gives

$$
\operatorname{Ent}_{\mu\otimes\nu}(f^2)
=\int\operatorname{Ent}_\nu(f(x,\cdot)^2)\mu(dx)
+\operatorname{Ent}_\mu(g^2).
$$

Cauchy-Schwarz implies

$$
|\nabla_xg|^2
=\frac{\left|\int f\nabla_x f\,d\nu\right|^2}{\int f^2d\nu}
\leq\int|\nabla_xf|^2d\nu,
$$

with the zero-denominator case handled by regularization. Apply the two LSIs and iterate. The resulting bound is

$$
\operatorname{Ent}_\pi(f^2)
\leq2\sum_i C_i\int|\nabla_i f|^2d\pi,
$$

which proves (15.6). The product structure is required of the reference measure, not of the evolving density.
:::

### 2.3. Bounded perturbations and nonconvex potentials

:::{prf:theorem} Bounded-density perturbation of an LSI
:label: thm-lsi-perturbation

Suppose $\mu$ satisfies (15.1) with constant $C_0$ and $\pi=w\mu$, where

$$
0<a\leq w\leq b<\infty.
$$

Then

$$
C_{\mathrm{LSI}}(\pi)\leq\frac ba C_0.
\tag{15.7}
$$

In particular, if $d\pi=Z_B^{-1}e^{-B}d\mu$ and $\operatorname{osc}(B)=\sup B-\inf B<\infty$, then

$$
C_{\mathrm{LSI}}(\pi)\leq e^{\operatorname{osc}(B)}C_0.
$$
:::

:::{prf:proof}
For $g\geq0$ and $c>0$, set $\psi_c(g)=g\log(g/c)-g+c\geq0$. Minimizing over $c$ gives

$$
\operatorname{Ent}_\pi(g)=\inf_{c>0}\int\psi_c(g)d\pi.
$$

Choose $c=\mu(g)$. Then

$$
\operatorname{Ent}_\pi(f^2)
\leq b\operatorname{Ent}_\mu(f^2)
\leq2b C_0\int|\nabla f|^2d\mu
\leq2\frac ba C_0\int|\nabla f|^2d\pi.
$$

For the exponential tilt, the ratio of the upper and lower density bounds is $e^{\operatorname{osc}(B)}$; its normalization cancels.
:::

### 2.4. Entropy, transport, and quadratic confinement

:::{prf:theorem} Entropy-transport-Fisher inequality
:label: thm-hwi-inequality

Let $\pi(dx)=Z^{-1}e^{-V(x)}dx$ have finite second moment and $\nabla^2V\succeq-KI$ for $K\geq0$. For a probability law $\mu$ with finite relative Fisher information and finite second moment,

$$
D_{\mathrm{KL}}(\mu\Vert\pi)
\leq W_2(\mu,\pi)\sqrt{I(\mu\Vert\pi)}
+\frac K2W_2^2(\mu,\pi).
$$

For $\nabla^2V\succeq\kappa I$ with $\kappa\geq0$, the final term is instead $-\kappa W_2^2/2$.
:::

:::{prf:proof}
First consider smooth densities and a regular optimal gradient transport $T$ from $\mu$ to $\pi$. Put $T_s=(1-s)\operatorname{id}+sT$ and $\mu_s=(T_s)_\#\mu$. The change-of-variables formula expresses the entropy as the sum of an internal term containing
$-\log\det((1-s)I+sDT)$ and a potential term $\int V(T_s(x))\mu(dx)$. The internal term is convex in $s$, because $DT$ is positive semidefinite and $-\log\det$ is convex. The second derivative of the potential term is at least $-K\int|T-x|^2d\mu$.

Thus, with $H(s)=D_{\mathrm{KL}}(\mu_s\Vert\pi)$,

$$
0=H(1)\geq H(0)+H'(0)-\frac K2W_2^2(\mu,\pi).
$$

The continuity equation for the interpolation gives

$$
H'(0)=\int\nabla\log(d\mu/d\pi)\cdot(T-x)\,d\mu.
$$

Cauchy-Schwarz proves the claim. A positive lower curvature bound gives the stated negative term by the same calculation. The usual smooth approximation of the transport interpolation extends the inequality to the finite-information domain; equivalently, the argument is the integrated above-tangent inequality for entropy.
:::

:::{prf:lemma} Weighted estimate from an elliptic Lyapunov function
:label: lem-kl-lyapunov-weighted-energy

Let $\mathscr L=\Delta-\nabla V\cdot\nabla$ preserve $\pi$, and suppose a smooth positive function $W$ satisfies

$$
\frac{\mathscr LW}{W}\leq-c|x|^2+b,
\qquad c>0,\quad b>0.
$$

Then

$$
c\int|x|^2f^2d\pi
\leq\int|\nabla f|^2d\pi+b\int f^2d\pi.
$$

In particular, $\pi(|x|^2)\leq b/c$.
:::

:::{prf:proof}
For compactly supported smooth $f$, integration by parts and completion of the square give

$$
\begin{aligned}
\int-\frac{\mathscr LW}{W}f^2d\pi
&=\int\left(2f\nabla\log W\cdot\nabla f
-f^2|\nabla\log W|^2\right)d\pi\\
&\leq\int|\nabla f|^2d\pi.
\end{aligned}
$$

Insert the Lyapunov inequality. Cutoff approximation extends the estimate to its form domain; taking cutoffs increasing to one gives the moment bound by Fatou's lemma.
:::

:::{prf:theorem} LSI from a quadratic Lyapunov bound
:label: thm-unconditional-lsi

Let $\pi(dx)=Z^{-1}e^{-V(x)}dx$ on $\mathbb R^d$, with $V\in C^2$ and $\nabla^2V\succeq-KI$ for some finite $K\geq0$. Suppose the Lyapunov bound of {prf:ref}`lem-kl-lyapunov-weighted-energy` holds. Then $\pi$ satisfies a full-gradient LSI, without a convexity assumption on $V$.

One finite constant can be constructed as follows. Choose $R$ with $\ell=cR^2-b>0$, and let $C_{\mathrm{loc}}$ be a Poincaré constant for $\pi$ conditioned on the ball $B_R$. Define

$$
C_P=\left(1+\frac b\ell\right)C_{\mathrm{loc}}+\frac1\ell,
\quad A_0=\frac1{2c},\quad B_0=\frac{4b}{c}.
$$

For any $\varepsilon>0$, put

$$
A_1=\sqrt{A_0}+\varepsilon+\frac K2A_0,
\qquad
B_1=\frac{B_0}{4\varepsilon}+\frac K2B_0.
$$

Then

$$
C_{\mathrm{LSI}}\leq
\frac{4A_1+(B_1+2)C_P}{2}.
$$
:::

:::{prf:proof}
**Poincaré inequality.** For a smooth $f$, subtract its mean on $B_R$ and call the resulting function $g$. Since $\operatorname{Var}_\pi(f)\leq\pi(g^2)$, it suffices to bound $\pi(g^2)$. The weighted estimate gives

$$
\ell\int_{B_R^c}g^2d\pi
\leq\int|\nabla g|^2d\pi+b\int_{B_R}g^2d\pi.
$$

The local Poincaré inequality controls the last integral. Adding the inside and outside contributions yields
$\operatorname{Var}_\pi(f)\leq C_P\int|\nabla f|^2d\pi$.
The constant $C_{\mathrm{loc}}$ is finite: the density has positive upper and lower bounds on the ball, so the usual Poincaré inequality on a ball transfers by comparison of measures.

**Defective entropy inequality.** For $\mu=h\pi$, apply the weighted estimate to $\sqrt h$:

$$
\int|x|^2d\mu\leq\frac{I(\mu\Vert\pi)}{4c}+\frac bc.
$$

The independent coupling of $\mu$ and $\pi$ gives

$$
W_2^2(\mu,\pi)
\leq2\mu(|x|^2)+2\pi(|x|^2)
\leq A_0I(\mu\Vert\pi)+B_0.
$$

Substitute into {prf:ref}`thm-hwi-inequality` and use
$\sqrt{B_0I}\leq\varepsilon I+B_0/(4\varepsilon)$. It follows that

$$
H_\pi(h)\leq A_1I_\pi(h)+B_1.
$$

By homogeneity this is
$\operatorname{Ent}_\pi(f^2)\leq4A_1\int|\nabla f|^2d\pi+B_1\pi(f^2)$.

**Removal of the defect.** The classical Rothaus centering inequality states

$$
\operatorname{Ent}_\pi(f^2)
\leq\operatorname{Ent}_\pi((f-\pi(f))^2)
+2\operatorname{Var}_\pi(f).
$$

Apply the defective inequality to $f-\pi(f)$, then use the Poincaré bound already proved. This gives

$$
\operatorname{Ent}_\pi(f^2)
\leq[4A_1+(B_1+2)C_P]\int|\nabla f|^2d\pi,
$$

which is the claimed LSI. The transport-and-centering argument and its classical ingredients are also given in [Cattiaux, Guillin, and Wu, Section 3.3](https://perso.math.univ-toulouse.fr/cattiaux/files/2013/11/cgw-submit-ptrf.pdf).
:::

:::{prf:corollary} The confining potential supplies the Lyapunov function
:label: thm-nonconvex-main

Suppose $U\in C^2(\mathbb R^d)$ satisfies

$$
x\cdot\nabla U(x)\geq\alpha_U|x|^2-b_U,
\qquad \alpha_U>0,\quad b_U\geq0,
\qquad \nabla^2U\succeq-M_-I.
$$

Then the spatial Gibbs law proportional to $e^{-U/\theta}$ has an LSI constant $C_x<\infty$. In {prf:ref}`thm-unconditional-lsi`, one may take

$$
s=\frac{\alpha_U}{4\theta},\qquad
W=e^{s|x|^2},\qquad
c=\frac{\alpha_U^2}{4\theta^2},\qquad
b=2sd+\frac{2sb_U}{\theta},\qquad K=M_-/\theta.
$$

A radial coercivity bound holding outside a ball gives the displayed global bound after increasing $b_U$ to control the compact interior.
:::

:::{prf:proof}
Radial integration of the coercivity inequality gives quadratic lower growth of $U$, up to a logarithmic term and a bounded interior contribution. Hence the Gibbs law is normalizable. For $\mathscr L=\Delta-\theta^{-1}\nabla U\cdot\nabla$,

$$
\frac{\mathscr LW}{W}
=2sd+4s^2|x|^2-\frac{2s}{\theta}x\cdot\nabla U
\leq b-c|x|^2.
$$

The stated $s$ gives the stated positive $c$. Apply {prf:ref}`thm-unconditional-lsi`. All constants concern the one-particle spatial law; the local Poincaré constant retains dependence on the interior landscape.
:::

:::{div} feynman-prose
The Lyapunov estimate prevents probability from escaping to infinity. The local Poincaré estimate controls the remaining bounded region, including any barriers between wells. Both enter the constant. This is why confinement can replace global convexity, while the depth and shape of the interior wells still affect the rate.
:::

### 2.5. The kinetic reference and its products

:::{prf:definition} Kinetic Gibbs reference measure
:label: def-gibbs-kinetic

Write

$$
\theta=\frac{\sigma_v^2}{2\gamma},
\qquad D=\frac{\sigma_v^2}{2}=\gamma\theta,
$$

and define

$$
m_U(dx,dv)=Z_U^{-1}
\exp\!\left[-\frac{U(x)+|v|^2/2}{\theta}\right]dx\,dv.
\tag{15.8}
$$

This is the invariant law of the conservative kinetic Langevin dynamics with potential $U$. It is a reference law for the entropy estimates, and is not by definition the invariant or quasi-stationary law after cloning is added.
:::

:::{prf:theorem} LSI for the kinetic reference measure
:label: thm-kinetic-lsi

Suppose the spatial Gibbs law has LSI constant $C_x$, as proved by either the bounded-perturbation argument or {prf:ref}`thm-nonconvex-main`. Then $m_U$ satisfies a full-gradient LSI with

$$
C_0=\max\{C_x,\theta\}.
\tag{15.9}
$$

For $U=W+B$ with $\nabla^2W\succeq\kappa I_d$ and bounded $\operatorname{osc}(B)$, one may take $C_x=\theta e^{\operatorname{osc}(B)/\theta}/\kappa$. In particular, the conclusion allows nonconvex $U$ and applies on $\mathbb R^d\times\mathbb R^d$.
:::

:::{prf:proof}
For the bounded-perturbation case, apply {prf:ref}`thm-bakry-emery` to $W/\theta$, then {prf:ref}`thm-lsi-perturbation` to its tilt by $e^{-B/\theta}$. For the confinement case, use the proof of {prf:ref}`thm-nonconvex-main`. The Gaussian velocity law $\mathcal N(0,\theta I_d)$ has constant $\theta$. Tensorization proves (15.9).
:::

:::{prf:corollary} Product kinetic LSI independent of population size
:label: cor-n-particle-kinetic-lsi

For every $N\geq1$, the reference law $m_U^{\otimes N}$ satisfies (15.1) with the same $C_0$ from (15.9).
:::

:::{prf:proof}
Apply {prf:ref}`thm-tensorization` to the $N$ copies of $m_U$.
:::

:::{div} feynman-prose
The distinction between a one-particle perturbation and a whole-swarm perturbation matters. A bounded nonconvex feature in each particle's potential is handled before tensorization, so its cost is paid once in the constant. If instead we compare the entire swarm density to a product density in one step, the oscillation of the whole logarithmic density ratio is the quantity we must bound.
:::

:::{prf:corollary} N-uniform LSI for specified joint laws
:label: cor-n-uniform-lsi

Let $\pi_N$ be a family of continuous joint probability laws on $\mathbb R^{2dN}$. Each of the following gives an $N$-uniform full-gradient LSI:

1. **Product reference:** $\pi_N=m_U^{\otimes N}$ under {prf:ref}`thm-kinetic-lsi`; take $C_*=C_0$.
2. **Bounded joint tilt:** $d\pi_N=Z_N^{-1}e^{-B_N}dm_U^{\otimes N}$ with $\sup_N\operatorname{osc}(B_N)\leq B_*<\infty$; take $C_*=e^{B_*}C_0$.
3. **Joint curvature:** $d\pi_N=Z_N^{-1}e^{-V_N}dS$ with $\nabla^2V_N\succeq\rho_* I_{2dN}$ uniformly in $N$, for $\rho_*>0$; take $C_*=1/\rho_*$.

4. **Contractive additive-noise flow:** $\pi_N$ is the invariant law in {prf:ref}`thm-kl-contractive-diffusion-lsi`, with $\lambda_{\min}(Q_N)\geq q_*>0$, $r_N\geq r_*>0$, and $c_{Q_N}\leq c_*<\infty$; take $C_*=2c_*/(r_*q_*)$.

Thus, in the respective cases,

$$
\operatorname{Ent}_{\pi_N}(f^2)
\leq2C_*\int\sum_{i=1}^N
\left(|\nabla_{x_i}f|^2+|\nabla_{v_i}f|^2\right)d\pi_N.
\tag{15.10}
$$

This conclusion applies to a swarm invariant law or a QSD when that particular law satisfies one of the stated structural hypotheses. For a law carrying discrete status variables, (15.3) specifies the additional entropy term.
:::

:::{prf:proof}
The first case is {prf:ref}`cor-n-particle-kinetic-lsi`. Apply (15.7) to obtain the second. Apply {prf:ref}`thm-bakry-emery` in dimension $2dN$ to obtain the third. The fourth follows from the flow and entropy-interpolation proof of {prf:ref}`thm-kl-contractive-diffusion-lsi`. None of these proofs introduces a factor of $N$ in the constant.
:::

:::{prf:proposition} A normalized interaction with an explicit curvature bound
:label: prop-kl-joint-interaction-curvature

Let

$$
V_N(z_1,\ldots,z_N)
=\sum_i V_0(z_i)+\frac{\varepsilon}{N}\sum_{i<j}W(z_i,z_j),
$$

where $\nabla^2V_0\succeq\rho_0I$, $\varepsilon\geq0$, and the Hessian of $W$ in both variables satisfies $\|\nabla^2 W\|_{\mathrm{op}}\leq L_W$. If $\varepsilon L_W<\rho_0$, then

$$
\nabla^2V_N\succeq(\rho_0-\varepsilon L_W)I,
\qquad
C_{\mathrm{LSI}}\leq\frac1{\rho_0-\varepsilon L_W},
\tag{15.11}
$$

uniformly in $N$.
:::

:::{prf:proof}
For $\xi=(\xi_1,\ldots,\xi_N)$,

$$
\frac{\varepsilon}{N}\sum_{i<j}
\begin{pmatrix}\xi_i\\\xi_j\end{pmatrix}^{\mathsf T}
\nabla^2W(z_i,z_j)
\begin{pmatrix}\xi_i\\\xi_j\end{pmatrix}
\geq-\varepsilon L_W\frac{N-1}{N}\sum_i|\xi_i|^2.
$$

Add the lower bound for $\sum_iV_0$ and apply the curvature case of (15.10).
:::

:::{prf:remark} Identifying the law in the static estimate
:label: rem-observation-composition-failure

Stationarity of the kinetic Gibbs law does not imply stationarity under cloning. For a conservative full generator $L=L_{\mathrm{kin}}+J$, a proposed density $\pi$ must satisfy $L^*\pi=0$. For a killed generator $A$, a proposed QSD must satisfy $A^*\nu=-\lambda\nu$, together with the relevant boundary conditions.

Exchangeability or propagation of chaos does not supply either density equation or an $N$-uniform bound on the joint density ratio. In particular, $\operatorname{osc}(B_N)\leq Nb$ in the second case of (15.10) yields the bound $e^{Nb}C_0$, not an $N$-uniform constant.
:::

(sec-fg-kl-conv-hypocoercivity)=
## 3. The kinetic entropy calculation

### 3.1. The modified functional

:::{prf:definition} Hypocoercive quadratic form
:label: def-hypocoercive-metric

For constants $a,c>0$ and $ac>b^2$, let

$$
G=\begin{pmatrix}cI&bI\\bI&aI\end{pmatrix},
\qquad
q=\begin{pmatrix}\nabla_x\log h\\\nabla_v\log h\end{pmatrix}.
$$

Define

$$
I_G(h)=\int h q^{\mathsf T}Gq\,d\pi
=cI_x(h)+2bI_{xv}(h)+aI_v(h),
$$

$$
I_{xv}(h)=\int h\nabla_x\log h\cdot\nabla_v\log h\,d\pi,
\qquad
\Phi_G(h)=H_\pi(h)+I_G(h).
\tag{15.12}
$$

If $g_-I\preceq G\preceq g_+I$, then

$$
g_-I_\pi(h)\leq I_G(h)\leq g_+I_\pi(h),
\qquad H_\pi(h)\leq\Phi_G(h).
$$

An LSI with constant $C_*$ consequently gives

$$
\Phi_G(h)\leq\left(C_*/2+g_+\right)I_\pi(h).
\tag{15.13}
$$
:::

:::{div} feynman-prose
The cross term records whether the position error and velocity error point in related directions. It may have either sign. The condition $ac>b^2$ ensures that the total gradient contribution remains positive. We will choose all three coefficients explicitly, so this positivity is part of the proof.
:::

### 3.2. Exact commutator identities

:::{prf:lemma} Kinetic Fisher-information identities
:label: lem-kinetic-evolution-bounds

Let $\pi=m_{U_N}$ be a kinetic Gibbs law on $\mathbb R^{2dN}$ with potential $U_N(x)$. Suppose $\|\nabla^2U_N\|_{\mathrm{op}}\leq M$ globally. For the relative density $h=f/\pi$, the forward kinetic equation becomes

$$
\partial_t h=\mathcal Kh,
\qquad
\mathcal K=D\Delta_v-\gamma v\cdot\nabla_v
-v\cdot\nabla_x+\nabla U_N\cdot\nabla_v.
\tag{15.14}
$$

For smooth positive solutions with justified integration by parts,

$$
\dot H=-DI_v,
\tag{15.15}
$$

and, writing $u=\log h$ and $H_U=\nabla^2U_N$,

$$
\begin{aligned}
\dot I_v
&=-2\gamma I_v-2I_{xv}
-2D\int h\|\nabla_v^2u\|_{\mathrm{HS}}^2d\pi,\\
\dot I_{xv}
&=-I_x-\gamma I_{xv}
+\int h(\nabla_vu)^{\mathsf T}H_U\nabla_vu\,d\pi\\
&\quad-2D\int h\sum_j
\partial_{v_j}\nabla_xu\cdot\partial_{v_j}\nabla_vu\,d\pi,\\
\dot I_x
&=2\int h(\nabla_xu)^{\mathsf T}H_U\nabla_vu\,d\pi
-2D\int h\|\nabla_v\nabla_xu\|_{\mathrm{HS}}^2d\pi.
\end{aligned}
\tag{15.16}
$$
:::

:::{prf:proof}
The fluctuation-dissipation relation $D=\gamma\theta$ and density (15.8) give (15.14) by expanding $\pi^{-1}L_{\mathrm{kin}}^*(\pi h)$. The velocity part is symmetric in $L^2(\pi)$, and the Hamiltonian part is antisymmetric. This proves (15.15).

The derivative commutators are

$$
[\nabla_v,\mathcal K]=-\nabla_x-\gamma\nabla_v,
\qquad
[\nabla_x,\mathcal K]=H_U\nabla_v.
$$

More explicitly, put

$$
B=\begin{pmatrix}0&H_U\\-I&-\gamma I\end{pmatrix}.
$$

Differentiating $I_G$ and integrating by parts gives

$$
\frac{d}{dt}I_G(h)
=-2D\int h\sum_j
(\partial_{v_j}q)^{\mathsf T}G(\partial_{v_j}q)d\pi
+\int h q^{\mathsf T}(GB+B^{\mathsf T}G)q\,d\pi.
\tag{15.17}
$$

To see the cancellation in this identity, use
$\partial_tu=\mathcal Ku+D|\nabla_vu|^2$ and
$\nabla\mathcal Ku=\mathcal Kq+Bq$ in the derivative of
$\int h q^{\mathsf T}Gq\,d\pi$.
The product rule and invariance replace the terms containing $\mathcal K$ by
$-2D\sum_j(\partial_{v_j}q)^{\mathsf T}G(\partial_{v_j}q)$;
the derivative of $D|\nabla_vu|^2$ cancels the corresponding first-gradient cross terms. The remaining drift terms are precisely $GB+B^{\mathsf T}G$.

Taking separately the velocity, cross, and position entries of (15.17) proves (15.16). The three second-derivative terms must be retained together: their combination is nonpositive because $G$ is positive definite.
:::

### 3.3. An explicit choice of coefficients

:::{prf:lemma} Dissipation of the modified entropy
:label: lem-hypocoercive-dissipation

Under {prf:ref}`lem-kinetic-evolution-bounds`, set

$$
L_M=2M+\gamma+2,
\qquad
\eta=\frac{D}{2(1+2M+L_M^2)},
\qquad a=c=2\eta,\quad b=\eta.
\tag{15.18}
$$

Then $G=\eta\begin{pmatrix}2I&I\\I&2I\end{pmatrix}$ has eigenvalues $\eta$ and $3\eta$, and

$$
\frac{d}{dt}\Phi_G(h_t)
\leq-\eta I_x(h_t)-\frac D2 I_v(h_t)
\leq-\eta I_\pi(h_t).
\tag{15.19}
$$
:::

:::{prf:proof}
The Hessian terms in (15.16) satisfy

$$
\int h(\nabla_vu)^{\mathsf T}H_U\nabla_vu\,d\pi\leq MI_v,
\qquad
\left|\int h(\nabla_xu)^{\mathsf T}H_U\nabla_vu\,d\pi\right|
\leq M\sqrt{I_xI_v}.
$$

Also $|I_{xv}|\leq\sqrt{I_xI_v}$. Combine (15.15)--(15.17), keeping the nonpositive second-derivative quadratic form, to obtain

$$
\dot\Phi_G
\leq-2\eta I_x
-\left(D+4\eta\gamma-2\eta M\right)I_v
+2\eta L_M\sqrt{I_xI_v}.
$$

Young's inequality in the form

$$
2\eta L_M\sqrt{I_xI_v}\leq\eta I_x+\eta L_M^2I_v
$$

therefore yields

$$
\dot\Phi_G
\leq-\eta I_x
-\left[D+4\eta\gamma-\eta(2M+L_M^2)\right]I_v.
$$

The choice (15.18) gives $\eta(2M+L_M^2)\leq D/2$ and $\eta\leq D/2$. This proves (15.19). In particular, neither the positive definiteness of $G$ nor the dissipation estimate uses an asymptotic approximation.
:::

:::{prf:theorem} Kinetic hypocoercive entropy convergence
:label: thm-villani-hypocoercivity

Suppose the kinetic Gibbs law $\pi=m_{U_N}$ satisfies a full-gradient LSI with constant $C_*$ and $\|\nabla^2U_N\|_{\mathrm{op}}\leq M$. Let $G$ and $\eta$ be given by (15.18). Then

$$
\Phi_G(h_t)\leq e^{-rt}\Phi_G(h_0),
\qquad
H_\pi(h_t)\leq e^{-rt}\Phi_G(h_0),
\qquad
r=\frac{\eta}{C_*/2+3\eta}>0.
\tag{15.20}
$$

The estimate holds for initial densities with finite $\Phi_G$, and after a positive time $t_0$ for any solution for which $\Phi_G(h_{t_0})<\infty$. If $C_*$ and $M$ are independent of $N$, then $r$ is independent of $N$.
:::

:::{prf:proof}
Combine (15.13) and (15.19) to obtain $\dot\Phi_G\leq-r\Phi_G$. Grönwall's inequality proves (15.20). Approximation extends the smooth calculation to finite-$\Phi_G$ data. Applying the same argument with initial time $t_0$ gives the positive-time statement.
:::

:::{prf:corollary} Nonconvex product potentials
:label: cor-n-particle-hypocoercive

Let $U_N(x)=\sum_iU(x_i)$, where $U$ satisfies {prf:ref}`thm-kinetic-lsi` and $\sup_x\|\nabla^2U(x)\|_{\mathrm{op}}\leq M$. Then (15.20) holds with $C_*=C_0$ from (15.9), uniformly in $N$. Global convexity of $U$ is not required.
:::

:::{prf:proof}
The full Hessian of $U_N$ is block diagonal with operator norm at most $M$. The reference law is $m_U^{\otimes N}$, whose LSI is {prf:ref}`cor-n-particle-kinetic-lsi`. Apply (15.20).
:::

:::{div} feynman-prose
This proof retains both pieces of the classical argument: an inequality for the equilibrium law and a dynamical calculation that transfers velocity dissipation to position. The first gives $C_*$; the second gives $\eta$. A small coefficient in front of Fisher information does not make that information bounded by entropy. That is why the initial factor in (15.20) is the modified entropy $\Phi_G(h_0)$.

The method is the entropic hypocoercivity construction described in [Villani's monograph](https://arxiv.org/abs/math/0609050). The calculation above supplies the coefficients and proof needed here.
:::

(sec-fg-kl-conv-full-generator)=
## 4. Cloning, killing, and the full evolution

### 4.1. The generator and its target law

:::{prf:definition} Full diffusion-jump generator
:label: def-kl-full-generator

On a continuous swarm state space, a conservative diffusion-jump realization has backward generator

$$
LF(S)=b_N(S)\cdot\nabla F(S)
+\operatorname{tr}\!\left(a_N(S)\nabla^2F(S)\right)
+\int_{E_N}[F(S')-F(S)]\,r_N(S,dS').
\tag{15.21}
$$

For the Euclidean kinetic part,

$$
b_N^{\mathrm{kin}}=(v_i,-\nabla U(x_i)-\gamma v_i)_{i=1}^N,
\qquad
a_N^{\mathrm{kin}}=D\,\operatorname{diag}(0,I)_{i=1}^N.
$$

The jump kernel $r_N$ includes the entire selected cloning update: companion sampling, acceptance, position jitter, and the coupled velocity update. It need not factorize over offspring. Additional geometric forces and diffusion enter $b_N$ and $a_N$ with the corresponding Itô correction when the model is specified in Stratonovich form.

With an interior killing rate $\kappa_N\geq0$, the killed generator is

$$
A=L-\kappa_N.
$$

For $\partial_t\widetilde f=A^*\widetilde f$ and $f=\widetilde f/\int\widetilde f$, on a domain without absorbing boundary flux,

$$
\partial_t f=L^*f-\kappa_Nf+\bar\kappa_f f,
\qquad
\bar\kappa_f=\int\kappa_N f.
\tag{15.22}
$$

A QSD density $\nu$ therefore solves

$$
L^*\nu=(\kappa_N-\lambda_N)\nu,
\qquad \lambda_N=\nu(\kappa_N).
\tag{15.23}
$$

For a discrete algorithm, the object to analyze is its actual sub-Markov one-step kernel $Q_\tau$. A diffusion-jump generator represents a separately specified continuous-time realization or a justified scaling limit; it is not an exact replacement of a finite BAOAB-and-cloning step.
:::

:::{div} feynman-prose
Conditioning changes the equation because probability is removed and the survivors are renormalized. The last term in (15.22) puts the total mass back to one. It is present even when the killed dynamics are linear. Calling the QSD an ordinary invariant measure would erase precisely this term.
:::

### 4.2. Exact entropy production, including normalization

:::{prf:proposition} Entropy identity for a conditioned diffusion-jump process
:label: prop-kl-conditioned-entropy

Let $A=L-\kappa$ be as in (15.21), with bounded nonnegative $\kappa$, no absorbing boundary flux, and a positive QSD density $\nu$ satisfying (15.23). Set $h=f/\nu$ for the normalized solution (15.22), and define

$$
\psi(s)=s\log s-s+1,
\qquad
\mathfrak b(a,b)=a\log(a/b)-a+b.
$$

Both functions are nonnegative on their domains. Put

$$
\begin{aligned}
\mathcal D_\nu(h)
&=\int\frac{(\nabla h)^{\mathsf T}a_N\nabla h}{h}\,d\nu\\
&\quad+\int\nu(dS)\int r_N(S,dS')\,
\mathfrak b\bigl(h(S),h(S')\bigr).
\end{aligned}
\tag{15.24}
$$

Then, whenever the quantities are finite and the generator identities are justified,

$$
\frac{d}{dt}H_\nu(h)
=-\mathcal D_\nu(h)
+\bar\kappa_f H_\nu(h)-\int\kappa\,\psi(h)d\nu.
\tag{15.25}
$$

Equivalently, the last two terms are

$$
\bar\kappa_f-\lambda_N
-\operatorname{Cov}_f(\kappa,\log h).
$$

In particular,

$$
\frac{d}{dt}H_\nu(h)
\leq-\mathcal D_\nu(h)
+\operatorname{osc}(\kappa)H_\nu(h).
\tag{15.26}
$$

For a conservative invariant law, $\kappa=0$, this reduces to $\dot H=-\mathcal D_\nu(h)$, without requiring reversibility.
:::

:::{prf:proof}
Mass preservation of the normalized equation gives

$$
\dot H_\nu(h)=\int fL\log h
-\int\kappa f\log h+\bar\kappa_f H_\nu(h).
$$

The diffusion chain rule and the jump identity give, pointwise,

$$
hL\log h
=Lh-\frac{(\nabla h)^{\mathsf T}a_N\nabla h}{h}
-\int r_N(S,dS')\mathfrak b\bigl(h(S),h(S')\bigr).
$$

Integrating and using (15.23),

$$
\int Lh\,d\nu
=\int h(\kappa-\lambda_N)d\nu
=\bar\kappa_f-\lambda_N.
$$

Thus

$$
\dot H_\nu(h)
=-\mathcal D_\nu(h)+\bar\kappa_f-\lambda_N
-\int\kappa h\log h\,d\nu+\bar\kappa_fH_\nu(h).
$$

Since $\int\kappa\psi(h)d\nu=\int\kappa h\log h\,d\nu-\bar\kappa_f+\lambda_N$, this is (15.25). Finally, $H_\nu(h)=\int\psi(h)d\nu$, $\psi\geq0$, and $\bar\kappa_f\leq\sup\kappa$ imply (15.26).
:::

:::{prf:proposition} Absorbing kinetic boundary
:label: prop-kl-boundary-entropy

For kinetic motion in a smooth spatial domain $\Omega$, let

$$
\partial_+E_N=\{(x,v):x\in\partial\Omega,\ v\cdot n(x)>0\}
$$

denote the outgoing boundary, with the analogous sum over particle boundary faces for a swarm. Impose the absorbing incoming trace on $f$ and $\nu$. Suppose the outgoing traces and the integrations below exist, and interior jumps do not cross the boundary except through an included killing term. Define

$$
\ell_f=\int\kappa f
+\int_{\partial_+E_N}(v\cdot n)f\,d\sigma\,dv.
$$

Then the normalized forward equation has source $\ell_f f$, and

$$
\begin{aligned}
\dot H_\nu(h)
&=-\mathcal D_\nu(h)+\ell_fH_\nu(h)
-\int\kappa\psi(h)d\nu\\
&\quad-\int_{\partial_+E_N}(v\cdot n)\nu\psi(h)\,d\sigma\,dv.
\end{aligned}
\tag{15.27}
$$

The outgoing trace of $h$ is $f/\nu$ wherever $\nu>0$; the expression is interpreted by its entropy extension otherwise.
:::

:::{prf:proof}
The mass equation gives $d\int\widetilde f/dt=-\ell_{\widetilde f}$. In the entropy calculation, the transport integration by parts contributes
$-\int_{\partial_+E_N}(v\cdot n)f\log h$.
The corresponding eigenmeasure identity is

$$
\int Lh\,d\nu
=\int\kappa h\,d\nu-\lambda_N
+\int_{\partial_+E_N}(v\cdot n)\nu h.
$$

Use $\lambda_N=\nu(\kappa)+\int_{\partial_+E_N}(v\cdot n)\nu$, and combine the three boundary terms into $-\int_{\partial_+E_N}(v\cdot n)\nu\psi(h)$. The interior terms are those of (15.25).
:::

:::{prf:remark} Boundary and normalization estimates
:label: rem-kl-killing-normalization

The boundary entropy term in (15.27) is nonnegative before its minus sign. The normalization term $\ell_fH_\nu(h)$ is also present. A moment bound on $f$ does not remove it or supply a uniform bound on the boundary flux. Soft killing, reflecting boundaries, and absorption at the validity boundary therefore require their respective equations.
:::

### 4.3. Cloning as a Markov update

:::{prf:lemma} Entropy dissipation for a common invariant cloning kernel
:label: thm-cloning-entropy-contraction

Let $P$ be a conservative Markov kernel with $\pi P=\pi$. Then

$$
D_{\mathrm{KL}}(\mu P\Vert\pi)
\leq D_{\mathrm{KL}}(\mu\Vert\pi).
\tag{15.28}
$$

If the stronger channel estimate
$D_{\mathrm{KL}}(\mu P\Vert\pi)\leq q_JD_{\mathrm{KL}}(\mu\Vert\pi)$ holds with $q_J<1$, a Poisson jump generator $J=\omega(P-I)$ dissipates entropy at least at rate $\omega(1-q_J)$.
:::

:::{prf:proof}
Construct the joint laws $\mu(dS)P(S,dS')$ and $\pi(dS)P(S,dS')$. Their relative entropy is $D_{\mathrm{KL}}(\mu\Vert\pi)$. Marginalizing to $S'$ cannot increase relative entropy, and the second marginal is $\pi$. This proves (15.28).

For the last assertion, convexity of relative entropy gives

$$
D H_\pi(h)[P^\dagger h-h]
\leq H_\pi(P^\dagger h)-H_\pi(h)
\leq-(1-q_J)H_\pi(h),
$$

where $P^\dagger$ is the density operator relative to $\pi$. Multiply by $\omega$.
:::

:::{prf:lemma} Cloning contribution to the modified functional
:label: lem-cloning-gamma2-bound

Under the common-invariant-law hypothesis of (15.28), suppose the cloning density operator satisfies

$$
I_G(P^\dagger h)\leq A_J I_G(h)
\tag{15.29}
$$

for every density in the form domain. Then the generator $J=\omega(P-I)$ satisfies

$$
D\Phi_G(h)[\omega(P^\dagger h-h)]
\leq\omega(A_J-1)_+I_G(h).
\tag{15.30}
$$

Here (15.29) is a gradient estimate for the full kernel on its stated domain, rather than a consequence of a status-distance contraction.
:::

:::{prf:proof}
The map $(s,p)\mapsto p^{\mathsf T}Gp/s$ is convex for $s>0$, so $I_G(h)=\int(\nabla h)^{\mathsf T}G\nabla h/h\,d\pi$ is convex. Consequently,

$$
DI_G(h)[P^\dagger h-h]
\leq I_G(P^\dagger h)-I_G(h)
\leq(A_J-1)I_G(h).
$$

Add the nonpositive entropy contribution from (15.28), and replace $A_J-1$ by its positive part.
:::

:::{prf:theorem} Kinetic and cloning convergence with a common target
:label: thm-main-kl-convergence

Suppose the conservative kinetic dynamics satisfy {prf:ref}`thm-villani-hypocoercivity` with target $\pi$ and coefficient $\eta$. Suppose also that $\pi P=\pi$ and (15.29) holds for the cloning kernel. If

$$
\delta_*:=\eta-3\eta\omega(A_J-1)_+>0,
$$

then the combined generator $L_{\mathrm{kin}}+\omega(P-I)$ satisfies

$$
H_\pi(h_t)\leq\Phi_G(h_t)
\leq\exp\!\left[-\frac{\delta_*t}{C_*/2+3\eta}\right]\Phi_G(h_0).
\tag{15.31}
$$

The rate is $N$-uniform when all the constants in these hypotheses are $N$-uniform.
:::

:::{prf:proof}
Add (15.19) and (15.30), and use $I_G\leq3\eta I_\pi$. This gives $\dot\Phi_G\leq-\delta_*I_\pi$. Close with (15.13) and apply Grönwall.
:::

:::{div} feynman-prose
The common-target condition can be checked by starting the system at the proposed target and applying each operator. If cloning changes that law, the separate contraction argument does not apply. We must then compute the derivative for the full stationary equation. This is particularly relevant for fitness selection: the kinetic and selection contributions can balance only after they are added.
:::

### 4.4. The law-dependent full-generator estimate

:::{prf:lemma} First variation of modified entropy
:label: lem-kl-functional-first-variation

Let $\nu$ be a fixed positive reference law and $G$ a constant positive definite matrix. For a differentiable density curve $h_t$, put $r_t=\partial_t h_t$ and $q_t=\nabla\log h_t$. Then

$$
\begin{aligned}
\frac{d}{dt}\Phi_G(h_t)
&=\int(1+\log h_t)r_t\,d\nu\\
&\quad+\int\left[
2q_t^{\mathsf T}G\nabla r_t
-r_t q_t^{\mathsf T}Gq_t
\right]d\nu.
\end{aligned}
\tag{15.32}
$$

For the normalized killed evolution, the quantity to substitute is exactly

$$
r_t=\nu^{-1}A^*(\nu h_t)+\ell_{f_t}h_t,
\tag{15.33}
$$

with $\ell_f=\bar\kappa_f$ in (15.22), and with the total killing flux in (15.27).
:::

:::{prf:proof}
Differentiate $h\log h$ and
$(\nabla h)^{\mathsf T}G\nabla h/h$ under the integral. The quotient rule gives the second line of (15.32). Equation (15.33) follows by differentiating the normalized killed density. No invariance of $\nu$ under the conservative part is used.
:::

:::{prf:theorem} Entropy convergence for the full normalized swarm evolution
:label: thm-kl-convergence-euclidean

Let $\nu_N$ be the QSD of the specified finite-particle killed dynamics, or let it be the invariant law of a specified conservative dynamics. This statement concerns a continuous state space. An entire law carrying discrete status variables requires the additional status entropy and form from (15.3).

Suppose that, on the solution class under consideration:

1. The actual law $\nu_N$ satisfies the full-gradient LSI (15.2) with constant $C_N$.
2. A constant positive definite matrix $G_N$ satisfies $G_N\preceq g_{+,N}I$.
3. The full derivative (15.32), with the full source (15.33), satisfies

   $$
   \frac{d}{dt}\Phi_{G_N}(h_t)
   \leq-\delta_N I_{\nu_N}(h_t),
   \qquad \delta_N>0.
   \tag{15.34}
   $$

Then

$$
D_{\mathrm{KL}}(\mu_t\Vert\nu_N)
\leq\Phi_{G_N}(h_t)
\leq e^{-r_Nt}\Phi_{G_N}(h_0),
\qquad
r_N=\frac{\delta_N}{C_N/2+g_{+,N}}.
\tag{15.35}
$$

An $N$-uniform conclusion follows when $C_N\leq C_*$, $g_{+,N}\leq g_*$, and $\delta_N\geq\delta_*>0$ uniformly in $N$. The first bound is supplied by {prf:ref}`cor-n-uniform-lsi` whenever the QSD itself satisfies one of its density hypotheses. The conservative kinetic and common-target cloning cases have the explicit dissipation proofs (15.19) and (15.31).
:::

:::{prf:proof}
The LSI and the upper matrix bound give

$$
\Phi_{G_N}(h_t)
\leq(C_N/2+g_{+,N})I_{\nu_N}(h_t).
$$

Substitute this into (15.34), apply Grönwall, and use $H_{\nu_N}\leq\Phi_{G_N}$. The stated uniform bounds give $r_N\geq\delta_* /(C_*/2+g_*)$.
:::

:::{prf:remark} What must be estimated for the full swarm law
:label: rem-kl-full-law-estimates

For an application beyond the explicitly proved cases, (15.34) requires the derivative of the actual cloning kernel and the killing normalization in the same reference law. Bounded fitness values do not by themselves bound derivatives of the kernel. A bound of the form $c\sqrt{I_x}+bI_x$ cannot be replaced by $CI_x$ near $I_x=0$; its square-root term must cancel in the full stationary calculation or remain as a forcing term.

The regularity estimates {prf:ref}`lem-variance-gradient`, {prf:ref}`lem-variance-hessian`, {prf:ref}`thm-c1-regularity`, and {prf:ref}`thm-c2-regularity` in {doc}`14_a_geometric_gas_c3_regularity` give derivative bounds for the normalized measurements and fitness field under their stated weighted-kernel hypotheses. They are inputs to the full kernel derivative calculation. They do not identify its stationary density or supply (15.34) without that calculation.
:::

(sec-fg-kl-conv-q-process)=
## 5. A second route through the conditioned process

:::{div} feynman-prose
There is another exact way to handle survival. Weight each state by its future survival amplitude, evolve a conservative transformed process, and undo the weight afterward. This replaces a nonlinear normalization in time by a change of measure. The cost of that change must be retained, especially when the number of particles grows.
:::

:::{prf:proposition} Doob transform and its invariant law
:label: prop-kl-doob-transform

Suppose the killed semigroup has a positive right eigenfunction $\eta_N$ and left eigenmeasure $\nu_N$ satisfying

$$
Q_t\eta_N=e^{-\lambda_Nt}\eta_N,
\qquad
\nu_NQ_t=e^{-\lambda_Nt}\nu_N,
\qquad \nu_N(\eta_N)=1.
$$

Define

$$
P_t^\eta F=\frac{e^{\lambda_Nt}}{\eta_N}Q_t(\eta_NF),
\qquad
\widehat\pi_N=\eta_N\nu_N.
\tag{15.36}
$$

Then $P_t^\eta$ is conservative and preserves $\widehat\pi_N$. Its generator is

$$
L^\eta F=\eta_N^{-1}A(\eta_NF)+\lambda_NF.
$$

For (15.21), this adds the diffusion drift $2a_N\nabla\log\eta_N$ and replaces the jump rate by

$$
r_N^\eta(S,dS')=\frac{\eta_N(S')}{\eta_N(S)}r_N(S,dS').
\tag{15.37}
$$

For the reweighting map $\mathcal R_w\mu=w\mu/\mu(w)$,

$$
\frac{\mu_0Q_t}{\mu_0Q_t1}
=\mathcal R_{1/\eta_N}\bigl[(\mathcal R_{\eta_N}\mu_0)P_t^\eta\bigr],
\qquad
\nu_N=\mathcal R_{1/\eta_N}\widehat\pi_N.
\tag{15.38}
$$
:::

:::{prf:proof}
The right eigenfunction identity gives $P_t^\eta1=1$. The left eigenmeasure identity gives

$$
\int P_t^\eta F\,d\widehat\pi_N
=e^{\lambda_Nt}\nu_NQ_t(\eta_NF)
=\nu_N(\eta_NF).
$$

Expanding $A(\eta_NF)$ proves the generator formula and (15.37); the zero-order terms cancel by the eigenfunction equation. Finally,

$$
(\mathcal R_{\eta_N}\mu_0)P_t^\eta(dS)
=\frac{e^{\lambda_Nt}}{\mu_0(\eta_N)}\eta_N(S)\mu_0Q_t(dS).
$$

Reweighting by $1/\eta_N$ proves (15.38).
:::

:::{prf:lemma} Relative entropy under bounded reweighting
:label: lem-kl-bounded-reweighting

If $0<m\leq w\leq M<\infty$, then

$$
D_{\mathrm{KL}}(\mathcal R_w\alpha\Vert\mathcal R_w\beta)
\leq\frac Mm D_{\mathrm{KL}}(\alpha\Vert\beta).
\tag{15.39}
$$
:::

:::{prf:proof}
Apply to both laws the same experiment: accept a point $S$ with probability $w(S)/M$. The relative entropy of the joint point-and-acceptance laws is $D_{\mathrm{KL}}(\alpha\Vert\beta)$. Its conditional chain rule is the sum of a nonnegative Bernoulli entropy and the two conditional entropies, weighted by the acceptance and rejection probabilities under $\alpha$. The accepted conditional laws are $\mathcal R_w\alpha$ and $\mathcal R_w\beta$. Their weight is $\alpha(w)/M\geq m/M$. Dropping the other nonnegative terms proves (15.39).
:::

:::{prf:theorem} Transfer of entropy convergence to the QSD
:label: thm-main-kl-final

Under (15.36), suppose $0<m_N\leq\eta_N\leq M_N<\infty$, and put $R_N=M_N/m_N$. If the transformed process satisfies

$$
H_{\widehat\pi_N}(\widehat h_t)
\leq e^{-rt}\Phi_G(\widehat h_0),
$$

then the original survival-conditioned process satisfies

$$
D_{\mathrm{KL}}(\mu_t\Vert\nu_N)
\leq R_N e^{-rt}\Phi_G(\widehat h_0),
\qquad
\widehat h_0=
\frac{d(\mathcal R_{\eta_N}\mu_0)}{d\widehat\pi_N}.
\tag{15.40}
$$

If a stronger estimate
$H_{\widehat\pi_N}(\widehat h_t)\leq C e^{-rt}H_{\widehat\pi_N}(\widehat h_0)$ is available, then

$$
D_{\mathrm{KL}}(\mu_t\Vert\nu_N)
\leq C R_N^2 e^{-rt}D_{\mathrm{KL}}(\mu_0\Vert\nu_N).
\tag{15.41}
$$
:::

:::{prf:proof}
Apply (15.39) with $w=1/\eta_N$ to (15.38), then use the transformed entropy estimate. For (15.41), apply (15.39) a second time with $w=\eta_N$ to the initial laws.
:::

:::{prf:remark} Uniformity of the survival change of measure
:label: rem-kl-doob-uniformity

The invariant law of the transformed process is $\widehat\pi_N=\eta_N\nu_N$, not $\nu_N$. Its LSI must be established for that law. The estimates (15.40)--(15.41) preserve population-uniform constants only when the reweighting costs are also uniform. On an absorbing domain the right eigenfunction may approach zero at the boundary, so the bounded-reweighting argument is not automatic; the direct balance (15.27) remains available.
:::

(sec-fg-kl-conv-curvature)=
## 6. Quadratic-form estimates and parameter conditions

:::{prf:definition} Auxiliary gradient form and iterated form
:label: def-hypo-carre-du-champ

For $\lambda>\mu^2$, define

$$
\mathcal Q(f)
=|\nabla_vf|^2+\lambda|\nabla_xf|^2
+2\mu\nabla_vf\cdot\nabla_xf.
$$

Let $\mathcal Q(f,g)$ be its polarization. For a linear generator $L$, define

$$
\mathcal Q_{2,L}(f)
=\frac12 L\mathcal Q(f)-\mathcal Q(f,Lf).
\tag{15.42}
$$

This is an auxiliary first-derivative form; it is distinct from the actual diffusion-jump carré du champ of $L$.
:::

:::{prf:lemma} Additivity in the generator
:label: lem-gamma2-decomposition

For $L=L_1+L_2$ and a fixed form $\mathcal Q$,

$$
\mathcal Q_{2,L}=\mathcal Q_{2,L_1}+\mathcal Q_{2,L_2}.
$$
:::

:::{prf:proof}
Expand (15.42), using linearity of $L$ and bilinearity of $\mathcal Q(f,g)$.
:::

:::{prf:theorem} Exact absorption of a spatial-gradient penalty
:label: thm-hypo-curvature-bound

Suppose estimates for the specified generators have established

$$
\mathcal Q_{2,L_{\mathrm{kin}}}(f)
\geq\alpha\mathcal Q(f)-\beta|\nabla_xf|^2,
\qquad
\mathcal Q_{2,J}(f)\geq-\varepsilon\mathcal Q(f),
$$

where $\beta\geq0$ and $\varepsilon\geq0$. Then

$$
\mathcal Q_{2,L_{\mathrm{kin}}+J}(f)
\geq\rho_{\mathcal Q}\mathcal Q(f),
\qquad
\rho_{\mathcal Q}
=\alpha-\varepsilon-\frac{\beta}{\lambda-\mu^2}.
\tag{15.43}
$$
:::

:::{prf:proof}
Complete the square:

$$
\mathcal Q(f)
=|\nabla_vf+\mu\nabla_xf|^2
+(\lambda-\mu^2)|\nabla_xf|^2.
$$

Hence $|\nabla_xf|^2\leq\mathcal Q(f)/(\lambda-\mu^2)$. Add the two assumed generator estimates and substitute this inequality. Because the spatial penalty has a minus sign, replacing $1/(\lambda-\mu^2)$ by the smaller $1/\lambda$ would not be a valid lower bound.
:::

:::{prf:corollary} Both constraints in a minimum must hold
:label: cor-acoustic-limit-explicit

If $\alpha=c_1\min\{\gamma,\alpha_U/\sigma_v^2\}$, define

$$
T_*:=\frac1{c_1}\left(
\frac{\beta}{\lambda-\mu^2}+\varepsilon
\right).
$$

Then $\rho_{\mathcal Q}>0$ is equivalent to the two simultaneous conditions

$$
\gamma>T_* ,
\qquad
\frac{\alpha_U}{\sigma_v^2}>T_*.
\tag{15.44}
$$
:::

:::{prf:proof}
For real numbers $a,b,T$, the inequality $\min\{a,b\}>T$ holds exactly when both $a>T$ and $b>T$.
:::

:::{prf:corollary} Uniformity of the absorbed coefficient
:label: cor-n-uniform-curvature

If the constants in (15.43) satisfy uniform bounds with

$$
\inf_N\left(
\alpha_N-\varepsilon_N
-\frac{\beta_N}{\lambda_N^{\mathrm{form}}-\mu_N^2}
\right)>0,
$$

then the resulting auxiliary-form bound is uniform in $N$. Here $\lambda_N^{\mathrm{form}}$ is the form parameter, distinct from the QSD killing eigenvalue.
:::

:::{prf:proof}
Take the infimum of the coefficient in (15.43).
:::

:::{prf:remark} Scope of the auxiliary curvature calculation
:label: rem-kl-curvature-scope

The conclusion (15.43) is an algebraic consequence of its two generator inequalities. It does not establish those inequalities. For a jump generator, (15.24) contains a nonlocal entropy remainder, and an estimate for $\mathcal Q_{2,J}(f)$ must be connected to the entropy/Fisher calculation for that kernel. The complete kinetic entropy calculation is (15.15)--(15.19); the common-invariant cloning estimate is (15.29)--(15.30).

A contraction in a distance counting alive/dead differences cannot provide a bound on spatial derivatives: that distance vanishes between different configurations with the same status pattern. Similarly, a global Lyapunov integral estimate is not a pointwise curvature inequality.
:::

(sec-fg-kl-conv-smoothing)=
## 7. What selection and Gaussian smoothing provide

### 7.1. Fisher information after Gaussian noise

:::{prf:lemma} Gaussian smoothing and absolute Fisher information
:label: lem-cloning-fisher-info

Let $X$ have any probability law on $\mathbb R^m$, let $Z\sim\mathcal N(0,\delta^2I_m)$ be independent, and let $p$ be the density of $Y=X+Z$. Then

$$
I_{\mathrm{abs}}(p):=\int p|\nabla\log p|^2
\leq\frac m{\delta^2}.
\tag{15.45}
$$

For a smooth positive reference density $\pi$,

$$
I(p\Vert\pi)
\leq\frac{2m}{\delta^2}
+2\int p|\nabla\log\pi|^2,
\tag{15.46}
$$

provided the last integral is finite.
:::

:::{prf:proof}
Differentiating the Gaussian convolution gives the score identity

$$
\nabla\log p(Y)
=-\delta^{-2}\mathbb E[Z\mid Y].
$$

Conditional Jensen yields

$$
I_{\mathrm{abs}}(p)
\leq\delta^{-4}\mathbb E|Z|^2=m/\delta^2.
$$

Apply $|a-b|^2\leq2|a|^2+2|b|^2$ to
$\nabla\log(p/\pi)=\nabla\log p-\nabla\log\pi$ to obtain (15.46).
:::

:::{prf:remark} Which coordinates have actually been smoothed
:label: rem-cloning-sublinear

The bounds (15.45)--(15.46) concern a full Gaussian convolution in the stated $m$ coordinates. They apply to a noisy offspring law in those coordinates. If some walkers are unchanged, or the velocity noise is coupled to conserve group momentum, the entire swarm update is not automatically such a convolution. A joint Fisher estimate must use the covariance and degeneracies of that actual update.

For $m=2dN$, the absolute Fisher bound is proportional to $N$. Dividing entropy and Fisher information by $N$ makes an intensive estimate; it does not by itself prove the joint LSI (15.10).
:::

### 7.2. Heat flow and its reference density

:::{prf:theorem} Relative entropy along heat flow
:label: thm-entropy-bound-debruijn

Let $\partial_tp_t=\tfrac12\Delta p_t$. For a fixed smooth positive reference density $q$, integrations by parts give

$$
\frac{d}{dt}D_{\mathrm{KL}}(p_t\Vert q)
=-\frac12 I_{\mathrm{abs}}(p_t)
-\frac12\int p_t\Delta\log q.
\tag{15.47}
$$

If the reference also solves $\partial_tq_t=\tfrac12\Delta q_t$, then

$$
\frac{d}{dt}D_{\mathrm{KL}}(p_t\Vert q_t)
=-\frac12 I(p_t\Vert q_t).
\tag{15.48}
$$

For a fixed reference $q$, the equation preserving it is instead

$$
\partial_tp=\frac12\nabla\cdot\left(p\nabla\log(p/q)\right),
$$

and this equation satisfies $\dot H=-I(p\Vert q)/2$. If $q$ has LSI constant $C_q$, its entropy therefore decays at least as $e^{-t/C_q}$.
:::

:::{prf:proof}
For fixed $q$,

$$
\dot H=\frac12\int\Delta p\log p
-\frac12\int\Delta p\log q,
$$

which is (15.47). For a moving heat-flow reference, there is the additional term
$-\frac12\int p\Delta q/q$. Use
$\Delta q/q=\Delta\log q+|\nabla\log q|^2$ and
$\int p\Delta\log q=-\int p\nabla\log p\cdot\nabla\log q$
to obtain the square in (15.48).

For the reference-preserving equation, integration by parts directly gives $\dot H=-I/2$. Combining this with $H\leq C_q I/2$ proves the final assertion.
:::

:::{prf:example} Smoothing changes a Gaussian target
:label: ex-kl-fixed-gaussian-smoothing

Take $p_0=q=\mathcal N(0,\theta I_m)$. Heat flow gives $p_t=\mathcal N(0,(\theta+t)I_m)$, so

$$
D_{\mathrm{KL}}(p_t\Vert q)
=\frac m2\left[\frac t\theta-\log\!\left(1+\frac t\theta\right)\right]>0
\quad(t>0).
$$

Thus Gaussian convolution does not contract relative entropy to an arbitrary fixed target, even when that target satisfies an LSI.
:::

### 7.3. A valid selection-energy calculation

:::{prf:proposition} Symmetrization for capped selection rates
:label: lem-meanfield-cloning-dissipation-hybrid

Let $\mu$ be a probability law and let an energy $E$ have oscillation at most $L<\infty$. Consider the specified pairwise selection mechanism with rate

$$
p(z,z')=\omega\min\{1,e^{-a(E(z')-E(z))}\},
\qquad a>0,
$$

which replaces a donor state $z$ by $z'$, with pairs distributed as $\mu\otimes\mu$. Before adding offspring noise, its contribution to mean energy is

$$
\begin{aligned}
\mathcal S_E(\mu)
&=\iint p(z,z')[E(z')-E(z)]\,\mu(dz)\mu(dz')\\
&=-\frac\omega2\iint
|E(z')-E(z)|\left(1-e^{-a|E(z')-E(z)|}\right)
\mu(dz)\mu(dz')\\
&\leq-\omega c_L\operatorname{Var}_\mu(E),
\end{aligned}
\tag{15.49}
$$

where $c_L=(1-e^{-aL})/L$ for $L>0$, with $c_0=a$.
:::

:::{prf:proof}
Swap $z$ and $z'$ in the full product integral and average. For $\Delta=E(z')-E(z)>0$,
$p(z,z')-p(z',z)=\omega(e^{-a\Delta}-1)$; for $\Delta<0$, use the opposite sign. This gives the equality in (15.49).

The function $s\mapsto(1-e^{-as})/s$ decreases on $(0,\infty)$ because $e^{as}\geq1+as$. Thus $1-e^{-as}\geq c_Ls$ for $0\leq s\leq L$. Finally,
$\iint(E(z')-E(z))^2\mu(dz)\mu(dz')=2\operatorname{Var}_\mu(E)$.
:::

:::{prf:remark} Energy descent and relative entropy
:label: rem-kl-selection-energy

The pair rate in (15.49) is stated explicitly. Other acceptance rules, sampled fitness values, or offspring noise require their own generator contribution. The symmetrization uses the full product integral; swapping variables also swaps a domain restricted by $E(z')>E(z)$.

The conclusion is an energy-variance estimate. It does not imply
$\operatorname{Var}_\mu(E)\geq cD_{\mathrm{KL}}(\mu\Vert\pi)$ for arbitrary $\mu$. For example, increasingly narrow smooth densities can have vanishing energy variance while their entropy relative to a fixed smooth $\pi$ diverges. Entropy convergence instead uses the full balances (15.25), (15.32), or a common-target Markov contraction.
:::

:::{div} feynman-prose
These calculations keep the useful pieces of the selection argument. Selection can lower a specified energy; Gaussian noise can bound the roughness of an offspring density. To prove convergence, those pieces must refer to the same evolution and target. Neither changing the target during smoothing nor replacing entropy by an energy variance supplies that connection.
:::

### 7.4. Stability of a normalized companion distribution

:::{prf:lemma} Companion-set perturbation with shared weights
:label: lem-softmax-lipschitz-status

Let $U_1,U_2$ be two candidate sets with identical positive weights $w_j$ on their intersection. Define $P_s(j)=w_j/Z_s$ on $U_s$, where $Z_s=\sum_{j\in U_s}w_j$. Put

$$
c=\sum_{j\in U_1\cap U_2}w_j,
\qquad
a=\sum_{j\in U_1\setminus U_2}w_j,
\qquad
b=\sum_{j\in U_2\setminus U_1}w_j.
$$

Then

$$
\|P_1-P_2\|_{\mathrm{TV}}
=\max\left\{\frac a{c+a},\frac b{c+b}\right\}.
$$

If $|U_s|\geq k$, $|U_1\triangle U_2|\leq n_c$, and $0<w_{\min}\leq w_j\leq w_{\max}$, then for any common bounded observable $F$,

$$
|P_1F-P_2F|
\leq2\|F\|_\infty\min\left\{1,
\frac{n_cw_{\max}}{kw_{\min}}\right\}.
$$
:::

:::{prf:proof}
Assume $a\geq b$, so $Z_1\geq Z_2$. Sum the absolute probability differences over the common set and the two disjoint parts:

$$
2\|P_1-P_2\|_{\mathrm{TV}}
=c\left(\frac1{c+b}-\frac1{c+a}\right)
+\frac a{c+a}+\frac b{c+b}
=\frac{2a}{c+a}.
$$

The other case is symmetric. Use $Z_s\geq kw_{\min}$ and $a,b\leq n_cw_{\max}$ for the final bound.
:::

:::{prf:remark} Candidate sets during sequential pairing
:label: rem-kl-companion-set-scope

This estimate is conditional on the candidate sets and on shared weights for common candidates. During sequential pairing, the current number of candidates may be much smaller than the original alive population. Moreover, moving common candidates changes their weights. Those effects require their own estimates before the lemma can be applied to a whole sequential matching law. For an unbounded Gaussian-distance kernel, a global positive $w_{\min}$ is not available; the exact formula in terms of $a,b,c$ remains valid.
:::

(sec-fg-kl-conv-discrete)=
## 8. Discrete time and the numerical kernel

:::{prf:definition} One-step entropy dissipation
:label: def-discrete-lsi

For a conservative Markov kernel $P$ preserving $\pi$, define

$$
\mathcal D_P(h)=H_\pi(h)-H_\pi(P^\dagger h).
$$

A one-step entropy inequality with coefficient $\varepsilon\in(0,1]$ is

$$
\mathcal D_P(h)\geq\varepsilon H_\pi(h).
\tag{15.50}
$$

This is an entropy-contraction inequality for the kernel. It is distinct from the static full-gradient LSI (15.1).
:::

:::{prf:definition} Markov-kernel quadratic form
:label: def-discrete-dirichlet

For $\pi P=\pi$, the quadratic form associated with $I-P$ is

$$
\mathcal E_P(f,f)
=\langle f,(I-P)f\rangle_\pi
=\frac12\int\pi(dS)P(S,dS')[f(S')-f(S)]^2.
$$

In general this is not $\int(f-Pf)^2d\pi$, and neither expression is the entropy dissipation $\mathcal D_P$.
:::

:::{prf:theorem} Iteration of one-step entropy contraction
:label: thm-lsi-implies-kl-convergence

Under (15.50),

$$
H_\pi((P^\dagger)^nh_0)
\leq(1-\varepsilon)^nH_\pi(h_0)
\leq e^{-\varepsilon n}H_\pi(h_0).
\tag{15.51}
$$

For a sub-Markov kernel $Q$, the same iteration applies if the inequality has instead been established directly for the normalized map $\mu\mapsto\mu Q/\mu Q1$ with target QSD $\nu$.
:::

:::{prf:proof}
Rearrange (15.50) to obtain the one-step bound and iterate. Use $1-\varepsilon\leq e^{-\varepsilon}$. The normalized map also iterates, since
$\mathcal T_Q^n\mu=\mu Q^n/\mu Q^n1$ whenever the denominators are positive.
:::

:::{prf:theorem} Composition of bounds with a common functional
:label: thm-main-lsi-composition

Let two updates $T_1,T_2$ satisfy, for the same nonnegative functional $\Phi$ and the same target law,

$$
\Phi(T_1\mu)\leq q_1\Phi(\mu)+b_1,
\qquad
\Phi(T_2\mu)\leq q_2\Phi(\mu)+b_2,
$$

where $q_1,q_2,b_1,b_2\geq0$. Then

$$
\Phi(T_2T_1\mu)
\leq q_1q_2\Phi(\mu)+q_2b_1+b_2.
\tag{15.52}
$$

If $q=q_1q_2<1$, iteration yields

$$
\Phi(\mu_n)
\leq q^n\Phi(\mu_0)
+\frac{q_2b_1+b_2}{1-q}(1-q^n).
\tag{15.53}
$$
:::

:::{prf:proof}
Substitute the first bound into the second and sum the geometric series. A nonzero additive term remains in (15.53); it cannot be removed to infer convergence to the target law.
:::

:::{prf:definition} Entropy-transport functional
:label: def-entropy-transport-lyapunov

For laws with finite second moments and a specified target $\pi$, put

$$
\mathcal V(\mu)=D_{\mathrm{KL}}(\mu\Vert\pi)+cW_2^2(\mu,\pi),
\qquad c>0.
$$
:::

:::{prf:theorem} Algebraic entropy-transport contraction
:label: thm-entropy-transport-contraction

Suppose one full update satisfies, with $H=D_{\mathrm{KL}}(\mu\Vert\pi)$ and $W=W_2(\mu,\pi)$,

$$
H'\leq aH-bW^2+d_H,
\qquad
(W')^2\leq KW^2+d_W,
$$

where $a,b,K,d_H,d_W\geq0$. Then

$$
\mathcal V(\mu')
\leq q\mathcal V(\mu)+d_H+cd_W,
\qquad q=\max\{a,K-b/c,0\}.
\tag{15.54}
$$

If $q<1$, the geometric-series conclusion (15.53) holds with additive term $d_H+cd_W$.
:::

:::{prf:proof}
Add $c$ times the second input inequality to the first:
$\mathcal V'\leq aH+(cK-b)W^2+d_H+cd_W$.
The definition of $q$ bounds the first two terms by $q(H+cW^2)$.
:::

:::{prf:remark} Inputs to an entropy-transport estimate
:label: rem-note-entropy-transport-innovation

The algebra in (15.54) retains a useful way of combining estimates. Its entropy input must be proved for the actual update. A contraction of $W_2$ alone does not imply the negative entropy term $-bW_2^2$. A linear mixture $(1-s)\mu+s\nu$ is also not generally a Wasserstein geodesic, so displacement convexity cannot be applied to that mixture as if it were the transport interpolation.
:::

:::{prf:lemma} Numerical entropy defect
:label: lem-discrete-lsi-from-curvature

Let an exact evolution $T_\tau$ satisfy
$\Phi(T_\tau\mu)\leq e^{-r\tau}\Phi(\mu)$.
Suppose the numerical update $\widetilde T_\tau$ has a functional error estimate

$$
\Phi(\widetilde T_\tau\mu)
\leq\Phi(T_\tau\mu)+K\tau^{p+1}\Phi(\mu)+B\tau^{p+1}
\tag{15.55}
$$

on an invariant class of input laws, with $p>0$, $K,B\geq0$. Set $q_\tau=e^{-r\tau}+K\tau^{p+1}$. For $q_\tau<1$,

$$
\Phi(\mu_n)
\leq q_\tau^n\Phi(\mu_0)
+\frac{B\tau^{p+1}}{1-q_\tau}(1-q_\tau^n).
\tag{15.56}
$$

For fixed $r>0$ and sufficiently small $\tau$, the second term is $O(\tau^p)$. If $B=0$, the estimate gives exact contraction of the functional to zero.
:::

:::{prf:proof}
Combine the exact bound and (15.55), then iterate the resulting affine recursion. Since $p>0$,
$1-q_\tau=r\tau+o(\tau)$ as $\tau\downarrow0$, which gives the stated order of the residual term.
:::

:::{prf:remark} What a BAOAB weak-error estimate does not supply
:label: rem-kl-discretization-domain

Relative entropy and Fisher information are nonlinear functionals of the evolving density. An error estimate for expectations of smooth observables is not, by itself, (15.55). That estimate needs density and derivative control in a class where the functional can be compared. The numerical invariant law may also differ from the continuous invariant law; the static LSI must then be transferred by an actual measure comparison such as (15.7), or proved directly for the numerical law.
:::

(sec-fg-kl-conv-consequences)=
## 9. Consequences and limiting laws

### 9.1. Marginals and empirical observables

:::{prf:corollary} Poincaré inequality and empirical observables
:label: cor-quantitative-lsi-final

If the continuous joint law $\pi_N$ satisfies (15.10), then

$$
\operatorname{Var}_{\pi_N}(F)
\leq C_*\int\sum_i
\left(|\nabla_{x_i}F|^2+|\nabla_{v_i}F|^2\right)d\pi_N.
\tag{15.57}
$$

In particular, for $F_N=N^{-1}\sum_i\varphi(z_i)$ with $\|\nabla\varphi\|_\infty\leq L$,

$$
\operatorname{Var}_{\pi_N}(F_N)\leq\frac{C_*L^2}{N}.
\tag{15.58}
$$

No independence of the coordinates is needed once the joint LSI has been established.
:::

:::{prf:proof}
For a bounded smooth mean-zero $F$, substitute $f=1+\varepsilon F$ into (15.10). The second-order entropy expansion is
$\operatorname{Ent}_{\pi_N}((1+\varepsilon F)^2)=2\varepsilon^2\pi_N(F^2)+o(\varepsilon^2)$.
Divide by $2\varepsilon^2$ and let $\varepsilon\to0$; approximation gives (15.57). For $F_N$, each gradient is $N^{-1}\nabla\varphi(z_i)$, so their squared sum is at most $L^2/N$.
:::

:::{prf:corollary} LSI for fixed marginals and their limits
:label: cor-kl-lsi-mean-field-limit

Suppose $\pi_N$ satisfies (15.10) uniformly. Its marginal on any fixed $k$ particles satisfies the same LSI constant $C_*$. If these marginals converge weakly to a law $\pi^{(k)}$, the limiting law satisfies the same inequality for smooth bounded test functions with bounded continuous squared gradient, and hence on the associated Sobolev closure when that class is a core.
:::

:::{prf:proof}
Apply (15.10) to a function depending only on the first $k$ particles; all other derivatives vanish. For the limit, apply the inequality to a fixed test function. Its entropy integrand and its squared gradient are bounded continuous functions, so both sides pass to the weak limit. Closure extends the inequality to the stated domain.
:::

:::{prf:remark} Marginal convergence, empirical measures, and KL
:label: rem-kl-tv-comparison

A finite empirical measure is atomic. Its relative entropy to a continuous positive density is therefore infinite. Entropy convergence in (15.35) concerns the law of the swarm, or a marginal obtained from that law. Smoothed empirical measures are different objects and can be studied using (15.45)--(15.47).

Relative entropy decreases under taking a marginal and implies total-variation control through Pinsker's inequality. Neither a Wasserstein convergence bound nor propagation of chaos supplies an upper bound on KL without an additional density estimate. KL also has no general triangle identity through a third probability measure.
:::

### 9.2. LSI for a non-Gibbs invariant law from flow contraction

:::{prf:theorem} Additive-noise diffusion with a contractive flow
:label: thm-kl-contractive-diffusion-lsi

Consider $dZ_t=b(Z_t)dt+\Sigma dW_t$ on $\mathbb R^m$, with constant $\Sigma$ and a globally Lipschitz $C^1$ drift. Suppose its synchronous flow satisfies

$$
|Z_t^z-Z_t^{z'}|_Q^2
\leq e^{-rt}|z-z'|_Q^2,
\qquad r>0,
$$

for a constant positive definite matrix $Q$, where $|z|_Q^2=z^{\mathsf T}Qz$. Let

$$
c_Q=\frac12\|Q^{1/2}\Sigma\|_{\mathrm{op}}^2.
$$

The diffusion has a unique invariant probability law $\pi$ with finite second moment, and this law satisfies

$$
\operatorname{Ent}_\pi(f^2)
\leq\frac{4c_Q}{r}\int
(\nabla f)^{\mathsf T}Q^{-1}\nabla f\,d\pi.
$$

Consequently its Euclidean full-gradient LSI constant is at most

$$
C_{\mathrm{LSI}}\leq\frac{2c_Q}{r\lambda_{\min}(Q)}.
$$
:::

:::{prf:proof}
**Invariant law.** Differentiating the contraction estimate at $t=0$ gives

$$
2\langle z-z',b(z)-b(z')\rangle_Q
\leq-r|z-z'|_Q^2.
$$

Set $z'=0$ and absorb the fixed vector $b(0)$ by Young's inequality. The generator then satisfies
$L|z|_Q^2\leq-(r/2)|z|_Q^2+C$, including the constant diffusion trace. Time-averaged laws are tight, and the Feller property gives an invariant law. Synchronous contraction gives its uniqueness and convergence to it in the quadratic transport distance induced by $Q$.

**Gradient estimate.** Let $J_t$ be the derivative of the flow in its initial condition. The contraction implies $J_t^{\mathsf T}QJ_t\preceq e^{-rt}Q$. For smooth positive $h$, differentiate $P_th$ through the flow and apply weighted Cauchy-Schwarz:

$$
\frac{(\nabla P_th)^{\mathsf T}Q^{-1}\nabla P_th}{P_th}
\leq e^{-rt}P_t\!\left(
\frac{(\nabla h)^{\mathsf T}Q^{-1}\nabla h}{h}
\right).
$$

Integrate against the invariant law to obtain
$I_{Q^{-1},\pi}(P_th)\leq e^{-rt}I_{Q^{-1},\pi}(h)$.

**Entropy interpolation.** The actual carré du champ is
$\Gamma_L(g,g)=|\Sigma^{\mathsf T}\nabla g|^2/2$ and satisfies
$\Gamma_L(g,g)\leq c_Q(\nabla g)^{\mathsf T}Q^{-1}\nabla g$.
For bounded positive $h$ bounded away from zero, invariance and the diffusion chain rule give

$$
H_\pi(h)
=\int_0^\infty\int\frac{\Gamma_L(P_th,P_th)}{P_th}\,d\pi\,dt
\leq\frac{c_Q}{r}I_{Q^{-1},\pi}(h).
$$

The entropy at infinite time vanishes by ergodicity. Substitute $h=f^2/\pi(f^2)$, then extend by approximation. The Euclidean bound follows from $Q^{-1}\preceq\lambda_{\min}(Q)^{-1}I$.
:::

:::{prf:corollary} Stationary LSI for a frozen alignment field
:label: cor-kl-frozen-alignment-lsi

Consider the conservative kinetic diffusion

$$
dX=Vdt,
\qquad
dV=[-\nabla U(X)-gV+\nu a(X)]dt+\sigma dW,
\qquad g=\gamma+\nu,
$$

where $m_UI\preceq\nabla^2U\preceq L_UI$, $m_U>0$, $\gamma>0$, $\nu\geq0$, and $a$ is a fixed $C^1$ Lipschitz field with $\|\nabla a\|_\infty\leq R$. Suppose

$$
A_x:=g(m_U-\nu R)-\frac{2(L_U+\nu R)^2}{g}>0.
$$

Set

$$
Q=\begin{pmatrix}(g^2/2)I&(g/2)I\\(g/2)I&I\end{pmatrix},
\qquad
r=\frac{\min\{A_x,g/2\}}{\lambda_{\max}(Q)}.
$$

Its invariant law $\pi^a$ satisfies a full-gradient LSI with

$$
C_{\mathrm{LSI}}(\pi^a)
\leq\frac{\sigma^2}{r\lambda_{\min}(Q)}.
$$

In particular, the result applies to a stationary mean-field law whose frozen alignment field satisfies these bounds and this strict parameter inequality. It does not require that $\pi^a$ have a Gibbs density.
:::

:::{prf:proof}
For two synchronously driven solutions, write $\xi=\Delta X$, $\zeta=\Delta V$, $\beta=g/2$, and $\alpha=g^2/2$. The noise cancels, and differentiation gives

$$
\begin{aligned}
\frac{d}{dt}(\alpha|\xi|^2+2\beta\xi\cdot\zeta+|\zeta|^2)
&\leq-g|\zeta|^2-g(m_U-\nu R)|\xi|^2\\
&\quad+2(L_U+\nu R)|\xi||\zeta|.
\end{aligned}
$$

Young's inequality bounds the last term by
$(g/2)|\zeta|^2+2(L_U+\nu R)^2|\xi|^2/g$.
Thus $d|\Delta Z|_Q^2/dt\leq-r|\Delta Z|_Q^2$.
The matrix is positive definite since $\alpha-\beta^2=g^2/4>0$. With $\Sigma=(0,\sigma I)^{\mathsf T}$, one has $c_Q=\sigma^2/2$, because the velocity block of $Q$ is $I$. Apply {prf:ref}`thm-kl-contractive-diffusion-lsi`.
:::

:::{div} feynman-prose
Here the invariant density need not be known explicitly. The replacement for a density formula is a checked contraction of the stochastic flow. Noise and that contraction give an inequality for the law through entropy interpolation. The inequality on $A_x$ retains both the friction requirement and the alignment penalty; small alignment alone does not make that coefficient positive for every friction value.
:::

### 9.3. Geometric forms and frozen velocity covariance

:::{prf:proposition} Frozen row-normalized alignment has a uniform Gaussian LSI
:label: prop-kl-frozen-ou-lsi

Fix positions $X$ and symmetric weights $K_{ij}=K_{ji}$, with $K_{ii}=0$ and $0<k_*\leq K_{ij}\leq k^*$ for $i\ne j$. For $N\geq2$, put

$$
d_i=\sum_{j\ne i}K_{ij},\qquad
D_X=\operatorname{diag}(d_i),\qquad
L_X=I-D_X^{-1}K,\qquad
A_X=\gamma I_{Nd}+\nu L_X\otimes I_d,
$$

where $\gamma>0$ and $\nu\geq0$. The frozen velocity equation

$$
dV_t=(-b_X-A_XV_t)dt+\sigma dW_t
$$

has stationary law $\mathcal N(-A_X^{-1}b_X,\Sigma_X)$, with

$$
A_X\Sigma_X+\Sigma_X A_X^{\mathsf T}=\sigma^2I,
\qquad
\|\Sigma_X\|_{\mathrm{op}}
\leq\frac{k^*}{k_*}\frac{\sigma^2}{2\gamma}.
$$

Its Gaussian LSI and Poincaré constants are at most this last bound, independently of $N$ and $\nu$.
:::

:::{prf:proof}
The matrix

$$
\widetilde L_X=D_X^{1/2}L_XD_X^{-1/2}
=D_X^{-1/2}(D_X-K)D_X^{-1/2}
$$

is symmetric positive semidefinite: for any $z$,
$z^{\mathsf T}(D_X-K)z=\frac12\sum_{i\ne j}K_{ij}(z_i-z_j)^2\geq0$.
The inequality $(z_i-z_j)^2\leq2z_i^2+2z_j^2$ also gives $\widetilde L_X\preceq2I$.

Set $H_X=D_X^{1/2}\otimes I_d$ and $\widetilde A_X=\gamma I+\nu\widetilde L_X\otimes I_d$. Then
$A_X=H_X^{-1}\widetilde A_XH_X$, with $\widetilde A_X\succeq\gamma I$. Consequently

$$
\|e^{-tA_X}\|_{\mathrm{op}}
\leq\sqrt{\frac{\max_i d_i}{\min_i d_i}}e^{-\gamma t}
\leq\sqrt{\frac{k^*}{k_*}}e^{-\gamma t}.
$$

Solving the linear equation gives the stationary mean and

$$
\Sigma_X=\sigma^2\int_0^\infty
e^{-tA_X}e^{-tA_X^{\mathsf T}}dt.
$$

The semigroup bound yields the covariance estimate; differentiating the integrand gives the Lyapunov equation. A Gaussian of covariance $\Sigma_X$ is a linear image of a standard Gaussian, so its LSI constant is at most $\|\Sigma_X\|_{\mathrm{op}}$; the Poincaré estimate follows by linearizing the LSI. The same argument works with a uniformly bounded degree ratio in place of the pointwise kernel bounds.
:::

:::{prf:corollary} Comparison with an adaptive geometric form
:label: cor-adaptive-lsi

Suppose a specified law $\pi_N$ satisfies (15.10), and let a full phase-space matrix field $G_N(S)$ satisfy

$$
G_N(S)\succeq g_*I_{2dN},
\qquad g_*>0,
$$

uniformly in $S,N$. Then

$$
\operatorname{Ent}_{\pi_N}(f^2)
\leq\frac{2C_*}{g_*}\int
(\nabla f)^{\mathsf T}G_N\nabla f\,d\pi_N.
\tag{15.59}
$$

If instead the law changes to $\widetilde\pi_N$ with uniformly bounded density ratio $a\leq d\widetilde\pi_N/d\pi_N\leq b$, the corresponding bound is $2(b/a)C_*/g_*$.
:::

:::{prf:proof}
The matrix inequality gives $|\nabla f|^2\leq g_*^{-1}(\nabla f)^{\mathsf T}G_N\nabla f$. Substitute this into the LSI. For the changed measure, first apply (15.7).
:::

:::{prf:remark} Frozen velocities and the full geometric law
:label: rem-kl-frozen-velocity-law

For positions held fixed, a linear velocity equation
$dV=-A(x)V\,dt+B(x)dW$ has Gaussian stationary covariance

$$
\Sigma_x=\int_0^\infty e^{-tA(x)}B(x)B(x)^{\mathsf T}e^{-tA(x)^{\mathsf T}}dt.
$$

If the symmetric part of $A(x)$ is at least $\gamma I$ and $\|B(x)\|_{\mathrm{op}}\leq b_*$, then
$\|\Sigma_x\|_{\mathrm{op}}\leq b_*^2/(2\gamma)$, by the bound $\|e^{-tA(x)}\|_{\mathrm{op}}\leq e^{-\gamma t}$. This gives a valid conditional Gaussian LSI for that frozen process.

It does not identify the conditional velocity law of the moving, cloning swarm. Furthermore, for any disintegration $\pi(dx,dv)=\pi_x(dx)\pi(dv\mid x)$,

$$
\operatorname{Var}_\pi(F)
=\mathbb E_{\pi_x}\operatorname{Var}_{\pi(\cdot\mid x)}(F)
+\operatorname{Var}_{\pi_x}\!\left(\mathbb E[F\mid x]\right).
$$

The second term requires spatial control. Uniform inequalities for the frozen Gaussian laws alone do not remove it. Likewise, row normalization of a graph Laplacian does not make it symmetric in the Euclidean inner product. The weighted norm in {prf:ref}`prop-kl-frozen-ou-lsi` provides the required comparison when its degree ratios are controlled.
:::

:::{div} feynman-prose
The population-independent constant becomes useful at (15.58): the gradient of an empirical average carries a factor $1/N$, while the sum over particles contributes $N$. The resulting variance is proportional to $1/N$. This conclusion is available for correlated walkers because the required control is already contained in the joint inequality.

For a limiting mean-field law, the order of the argument is equally concrete. Establish the inequality for the finite-particle laws, take a fixed marginal, then pass to its limit. Identifying that limit as the stationary solution of the mean-field equation uses the existence, uniqueness, and convergence results for that equation. A static LSI and a mean-field identification perform different jobs.
:::

(sec-fg-kl-conv-references-1)=
## 10. How the estimates fit together

:::{admonition} Constants and dependencies
:class: feynman-added note

| Estimate | Inputs | Result |
|---|---|---|
| Kinetic reference LSI | Quadratic radial confinement and a lower Hessian bound, or bounded perturbation of a convex potential | $C_0$ in (15.9) |
| Joint LSI | Product law, bounded joint tilt, uniform joint curvature, or uniform flow contraction | $C_*$ in (15.10) |
| Kinetic dissipation | Global Hessian bound $M$, $D=\sigma_v^2/2$, $\gamma>0$ | $\eta$ in (15.18) |
| Kinetic entropy convergence | Joint LSI and kinetic dissipation for the same Gibbs law | Rate (15.20) |
| Full conditioned evolution | Actual QSD LSI and the complete derivative estimate (15.34) | Rate (15.35) |
| Doob-transform route | Transformed-law estimate and bounded survival reweighting | (15.40)--(15.41) |
| Numerical iteration | A functional defect bound for the actual numerical kernel | (15.56) |
| Empirical variance | Full-gradient joint LSI | $C_*L^2/N$ in (15.58) |

The proofs of these implications are contained in this chapter. The operator-specific inputs come from the algorithm and convergence chapters, with the reference law, state space, and normalization fixed before they are combined.
:::

:::{div} feynman-prose
The essential calculation is now visible: an LSI controls entropy by full Fisher information, and a hypocoercive derivative controls the modified entropy by that same information. Killing adds a normalization term; cloning adds a kernel contribution. Once those contributions are estimated for the actual law, the final step is Grönwall's inequality. The resulting statement says exactly which distribution converges, which functional decays, and which constants remain independent of the population.
:::
