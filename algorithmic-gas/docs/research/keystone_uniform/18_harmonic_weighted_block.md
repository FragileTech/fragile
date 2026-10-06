# Harmonic weighted row coupling at the original step

This record proves the nonviscous kinetic part of the combined local and
global weighted-Hamming argument. Its reference kernel has the actual
harmonic force, both Gaussian innovations and the native smooth cap.
The dense count-viscous extension is a separate obligation below.

:::{prf:definition} The retained product kinetic kernel and combined cost
:label: def-kuhw-product-register

Let $t=h/2$, $c=e^{-\gamma h}$, $B_{\rm kin}=t(1+c)$,
$m=1-t^2$, and $a_x=1-tB_{\rm kin}$. Assume
$0<a_x<1$, $m>0$, $q,s,V>0$, and let $V_*\ge V$ be a declared
entering velocity envelope. For $F=-x$ and $\nu=0$, the exact one-row
kinetic kernel $K_0$ is
$$
\begin{aligned}
x^+&=a_xx+B_{\rm kin}v+tq\xi+s\chi,\\
v^+&=C_V(z_rx+z_vv+qm\xi),\\
z_r&=-t(c+a_x),\qquad z_v=c-tB_{\rm kin},\qquad
C_V(w)=\frac{Vw}{V+|w|}.
\end{aligned}
$$
Here $\xi,\chi$ are independent standard $d$-Gaussians.
For every fixed entering array, $K_{0,N}=K_0^{\otimes N}$ has independent
innovation rows in each own kernel. This is an exact description only
at $\nu=0$.

Put $W(z)=|x|^2+|v|^2$, $\mathsf M_2(S)=N^{-1}\sum_iW(z_i)$,
$I_i(S,T)=\mathbf1_{\{z_i\ne z_i'\}}$, and
$\rho_N=N^{-1}\sum_iI_i$. Define
$$
\begin{aligned}
D_{\rm loc}(S,T)
 &=\frac1N\sum_i I_i[1+\eta(W(z_i)+W(z_i'))],\\
D_{\eta,g}(S,T)
 &=D_{\rm loc}(S,T)
   +g\rho_N(S,T)[\mathsf M_2(S)+\mathsf M_2(T)],
\qquad \eta,g>0.
\end{aligned}
$$
The global term is not substituted for the local weighted term.
The combined cost is treated as a measurable distance-like cost;
no triangle inequality for it is asserted.
:::

:::{prf:lemma} Exact harmonic second-moment drift
:label: lem-kuhw-product-drift

Put
$$
\epsilon_Y=\frac{1-a_x^2}{2},\qquad
\lambda=\frac{1+a_x^2}{2}<1,\qquad \tau^2=t^2q^2+s^2,
$$
$$
B_2=(1+a_x^2/\epsilon_Y)B_{\rm kin}^2V_*^2
             +d\tau^2+V^2>0.
$$
For every entering row with $|v|\le V_*$,
$$
K_0W(z)\le\lambda W(z)+B_2.
$$
Consequently
$$
K_{0,N}\mathsf M_2(S)\le\lambda\mathsf M_2(S)+B_2
$$
for every $N$. For the iterated conservative product kernel on
$(\mathbb R^d\times\overline B_V)^N$, this gives
$$
\mathbb E\mathsf M_2(S_n)
\le\lambda^n\mathbb E\mathsf M_2(S_0)
       +B_2\frac{1-\lambda^n}{1-\lambda}.
$$
:::

:::{prf:proof}
The positional Gaussian has covariance $\tau^2I_d$ and zero mean.
Young's inequality gives
$$
|a_xx+B_{\rm kin}v|^2
\le(a_x^2+\epsilon_Y)|x|^2
 +(1+a_x^2/\epsilon_Y)B_{\rm kin}^2|v|^2.
$$
The actual cap ensures $|v^+|\le V$.
Hence the expectation of $W(z^+)$ is at most
$\lambda|x|^2+B_2\le\lambda W(z)+B_2$.
Sum rows and iterate the affine inequality. Neither Gaussian is
truncated and no position envelope is assumed. $\square$
:::

:::{prf:lemma} Explicit one-row common mass for the exact capped Gaussian law
:label: lem-kuhw-product-minorization

Choose $R_{\rm ref},r,u>0$ and put
$$
R=\frac{4B_2}{1-\lambda}+R_{\rm ref},\qquad
M_x=a_x\sqrt R+B_{\rm kin}V_*,
\qquad M_z=|z_r|\sqrt R+|z_v|V_*,
$$
$$
Q_\xi=\frac{u+M_z}{qm},\qquad
Q_\chi=\frac{r+M_x+tqQ_\xi}{s},
$$
$$
k_R=(2\pi)^{-d}(qms)^{-d}
       \exp[-(Q_\xi^2+Q_\chi^2)/2],\qquad
\varepsilon=\min\{1/2,v_d(r)v_d(u)k_R\}>0,
$$
where $v_d(b)=\pi^{d/2}b^d/\Gamma(1+d/2)$.
Let $\vartheta$ be uniform position on $B(0,r)$ times the actual
$C_V$-pushforward of uniform pre-cap velocity on $B(0,u)$.
Then
$$
K_0(z,\cdot)\ge\varepsilon\vartheta
\qquad\text{whenever } W(z)\le R,\quad |v|\le V_*.
$$
The coefficient has no population factor.
:::

:::{prf:proof}
Before applying the cap, let $X=x^+$ and
$Z=z_rx+z_vv+qm\xi$. The exact map $(\xi,\chi)\mapsto(Z,X)$
has determinant $(qms)^d$. For a target pair $|Z|\le u$, $|X|\le r$,
its unique latent coordinates obey
$$
|\xi|=\left|\frac{Z-z_rx-z_vv}{qm}\right|\le Q_\xi,
\qquad
|\chi|=\left|\frac{X-a_xx-B_{\rm kin}v-tq\xi}{s}\right|
\le Q_\chi.
$$
The joint density is therefore at least $k_R$ on the target product
of balls. Integrating that lower density and then pushing its
pre-cap velocity through the actual cap proves the assertion.
The two output coordinates share $\xi$; the joint change of variables
retains that correlation. $\square$
:::

:::{prf:theorem} Product harmonic contraction in the combined weighted cost
:label: thm-kuhw-product-combined

Use the preceding primitive constants and define
$$
\eta=\frac{\varepsilon}{2[R(1+\lambda)+2B_2]},\qquad
q_H=\max\left\{1-\varepsilon/2,\
\frac{1+\eta(\lambda R+2B_2)}{1+\eta R}\right\}<1,
$$
$$
g=\frac{1-q_H}{4B_2}>0,\qquad
q_D=\max\{(1+q_H)/2,\lambda\}<1.
$$
For every $N$ and every pair of entering arrays with velocities bounded
by $V_*$ there is a measurable coupling of their exact product kinetic
updates satisfying
$$
\rho_N(S^+,T^+)\le\rho_N(S,T)\quad\text{almost surely},
$$
$$
\mathbb E D_{\rm loc}(S^+,T^+)\le q_HD_{\rm loc}(S,T),
$$
$$
\boxed{\quad
\mathbb E D_{\eta,g}(S^+,T^+)
\le q_DD_{\eta,g}(S,T).
\quad}
$$
Thus the kinetic block length is explicitly one, and every contraction,
weight and moment constant is independent of $N$.
The primitive choice $h=0.04$, $\gamma=b_O=1$,
$q^2=(1-e^{-0.08})/2$, $s^2=0.1^2(0.04)$, $V=2$,
$V_*=4$, $d=3$, and $r=u=R_{\rm ref}=1$ satisfies all the hypotheses.
:::

:::{prf:proof}
For an equal entering row, use an identical draw from its common
kernel. Its outputs are equal with probability one.
For an unequal pair put $s_0=W(z)+W(z')$ and use the row cost
$$
d_\eta(z,z')=\mathbf1_{\{z\ne z'\}}[1+\eta(W(z)+W(z'))].
$$
If $s_0>R$, a product row coupling has expected cost at most
$1+\eta(\lambda s_0+2B_2)$. Its ratio to $1+\eta s_0$
decreases in $s_0$ and is at most the second endpoint in $q_H$.
This endpoint is below one because $R>2B_2/(1-\lambda)$.
If $s_0\le R$, both row laws dominate the same
$\varepsilon\vartheta$. Couple this common part identically and couple
the probability remainders arbitrarily. The exact row marginals are
preserved, and the expected cost is at most
$$
1-\varepsilon+\eta(\lambda s_0+2B_2)
\le1-\varepsilon/2,
$$
since $\eta(\lambda R+2B_2)\le\varepsilon/2$.
The input cost is at least one, so the first endpoint in $q_H$ applies.
Common-part and product couplings are measurable kernels: the
remainders are explicitly
$(K_0(z,\cdot)-\varepsilon\vartheta)/(1-\varepsilon)$
on the small set.

Apply these pair couplings independently across row labels.
Each own law is exactly $K_0^{\otimes N}$.
Every equal entering row remains equal, giving the pathwise
mismatch-count bound, and summing the row inequalities proves the
local weighted contraction.

For the global term use that same pathwise count bound before
expectation:
$$
\begin{aligned}
\mathbb E\{\rho_N(S^+,T^+)
 [\mathsf M_2(S^+)+\mathsf M_2(T^+)]\}
&\le\rho_N(S,T)
  \mathbb E[\mathsf M_2(S^+)+\mathsf M_2(T^+)]\\
&\le\rho_N(S,T)
 [\lambda(\mathsf M_2(S)+\mathsf M_2(T))+2B_2].
\end{aligned}
$$
No factorization of a reset event and an output moment is used.
Combining the local bound with this inequality and
$D_{\rm loc}\ge\rho_N$ yields
$$
\mathbb E D_{\eta,g}(S^+,T^+)
\le(q_H+2gB_2)D_{\rm loc}(S,T)
+\lambda g\rho_N(S,T)[\mathsf M_2(S)+\mathsf M_2(T)].
$$
The chosen $g$ gives $q_H+2gB_2=(1+q_H)/2$.
This proves the combined contraction.
At the displayed original parameters, $m,a_x,q,s$ are positive and
$a_x<1$. Thus $B_2,R,Q_\xi,Q_\chi$ are finite, $\varepsilon>0$,
$\eta,g>0$, and both contraction endpoints are strictly below one.
Their potentially small gaps are proved by exact formulas, without
floating-point subtraction. $\square$
:::

:::{prf:corollary} Invariant moment and both alive law endpoints
:label: cor-kuhw-product-invariant-alive

For the conservative product kernel on
$E_N=(\mathbb R^d\times\overline B_V)^N$ there is a unique invariant
law $\Pi_N$ with finite $\mathsf M_2$ moment. It is exactly
$\pi^{\otimes N}$, where $\pi$ is the one-row invariant law, and
$$
\pi W\le H_\infty:=\frac{B_2}{1-\lambda},\qquad
\int W(z_i)\,\Pi_N(dS)\le H_\infty\quad\text{for every }i,N.
$$
For any entering law $\mu_N$ with finite averaged moment, put
$$
A_{\mu_N}=1+(\eta+g)
 \left[\int\mathsf M_2(S)\,\mu_N(dS)+H_\infty\right].
$$
Let $\mathcal W_{2,N,G}$ be full-swarm Wasserstein distance with
squared ground cost $N^{-1}\sum_i|z_i-z_i'|_G^2$, for a positive
physical matrix $G$. Let $\mathcal E_N(S)=N^{-1}\sum_i\delta_{z_i}$,
and denote by $\mathcal W_{2,\mathrm{emp},G}$ Wasserstein distance
between random empirical-measure laws with ground distance $W_{2,G}$.
Then
$$
\begin{aligned}
\mathcal W_{2,N,G}(\mu_NK_{0,N}^n,\Pi_N)
&\le
\sqrt{\frac{2\lambda_{\max}(G)}{\eta}\,A_{\mu_N}}\ q_D^{n/2},\\
\mathcal W_{2,\mathrm{emp},G}
 ((\mathcal E_N)_\#(\mu_NK_{0,N}^n),(\mathcal E_N)_\#\Pi_N)
&\le
\sqrt{\frac{2\lambda_{\max}(G)}{\eta}\,A_{\mu_N}}\ q_D^{n/2},\\
W_{2,G}
 \left(\int\mathcal E_N\,d(\mu_NK_{0,N}^n),\,\pi\right)
&\le
\sqrt{\frac{2\lambda_{\max}(G)}{\eta}\,A_{\mu_N}}\ q_D^{n/2}.
\end{aligned}
$$
The sampled-alive total-variation distance is also at most
$A_{\mu_N}q_D^n$.
Thus all constants and both alive relaxation rates are independent
of $N$ when the initial averaged moment is bounded uniformly in $N$.
:::

:::{prf:proof}
For one-row probability laws, the optimal coupling cost associated
with $d_\eta$ is the weighted variation norm
$$
\inf_\Gamma\int d_\eta\,d\Gamma
=\int(1/2+\eta W)\,d|\mu-\widetilde\mu|.
$$
Indeed, couple the common part diagonally and the two mutually singular
remainders arbitrarily: their contributions are exactly the two
integrals of $1/2+\eta W$. Conversely, dual tests of absolute value
at most $1/2+\eta W$ bound every coupling cost below by that weighted
variation norm. The row coupling theorem therefore contracts this
norm by $q_H<1$.
Finite-$W$-moment probabilities form a closed subset of the complete
weighted signed-measure normed space: positivity and total mass
pass to a norm limit. The drift maps this subset into itself.
Successive iterates from a point mass have geometrically summable
norm differences, so their limit $\pi$ exists in the subset; the
contraction makes it invariant and unique. Integrating the drift
under $\pi$ gives $\pi W\le B_2/(1-\lambda)$.

The product $\Pi_N=\pi^{\otimes N}$ is invariant for the exact product
kernel. To establish uniqueness without a triangle inequality for
$D_{\eta,g}$, start with any two invariant laws of finite averaged
moment and any coupling between them. Its $D_{\eta,g}$ expectation
is finite because
$$
D_{\eta,g}(S,T)\le
1+(\eta+g)[\mathsf M_2(S)+\mathsf M_2(T)].
$$
Iterate the measurable combined-cost coupling. The output law
marginals remain those two invariant laws, while its expected
mismatch fraction tends to zero at rate $q_D^n$.
The probability that the full arrays differ is at most
$N\rho_N$, so their full-array total-variation distance tends to
zero for this fixed $N$. They are equal. This uniqueness argument
does not assert a population-uniform full-array TV prefactor.

Couple $\mu_N$ to $\Pi_N$ initially, for example independently.
The preceding cost bound gives initial expectation at most
$A_{\mu_N}$. Iteration gives output cost expectation at most
$A_{\mu_N}q_D^n$. Pointwise,
$$
\frac1N\sum_i|z_i-z_i'|_G^2
\le\frac{2\lambda_{\max}(G)}N
       \sum_i I_i[W(z_i)+W(z_i')]
\le\frac{2\lambda_{\max}(G)}{\eta}D_{\rm loc}(S,T).
$$
This proves the full-swarm physical Wasserstein estimate.
Fixed-slot empirical matching bounds the empirical ground
$W_{2,G}^2$ by the same physical cost, so pushing this coupling
through $\mathcal E_N$ proves the random empirical-law bound.
Sampling the same independent uniform row label in its two arrays
gives the sampled-alive Wasserstein bound and the TV bound:
the two sampled states differ with probability exactly the expected
$\rho_N$, at most the expected $D_{\eta,g}$.
Under $\Pi_N$ that sampled state has law $\pi$. $\square$
:::

:::{prf:proposition} Exact interface for active preparation
:label: prop-kuhw-active-composition-interface

Suppose the actual complete all-alive preparation admits a measurable
coupling, with entering velocities bounded by $V$ and prepared velocities
bounded by $V_*$, for which a proved primitive factor $C_{\rm prep}$ obeys
$$
\mathbb E D_{\eta,g}(S^{\rm prep},T^{\rm prep})
\le C_{\rm prep}D_{\eta,g}(S,T),
\qquad C_{\rm prep}q_D<1.
$$
Then the unchanged active preparation followed by the nonviscous kinetic
update has the complete contraction
$$
\mathbb E D_{\eta,g}(S^+,T^+)
\le C_{\rm prep}q_DD_{\eta,g}(S,T).
$$
In particular, once a complete preparation register gives
$C_{\rm prep}(\theta)=1+O(\theta)$ as $\theta\downarrow0$, the strictly
positive kinetic gap supplies a nonempty active interval. An explicit
primitive exponent endpoint still requires the complete preparation
factor; the global weighted estimate alone is insufficient.
:::

:::{prf:proof}
First generate the actual coupled preparations. Conditional on their
complete arrays, apply the product kinetic coupling just proved.
The conditional own kinetic marginals are exactly the actual product
kinetic law; after averaging, all source and component correlations
from preparation remain present. The tower property gives the product
of the two proved factors. No endpoint law from a surrogate
preparation is substituted. $\square$
:::

:::{prf:lemma} An explicit primitive envelope for the actual preparation factor
:label: lem-kuhw-positive-preparation-register

Use the actual conservative, current-frame preparation of research
record 17, with continuous bounded raw reward of oscillation
$R_{\rm osc}$, comparison-feature diameter $D_*$, role-weight floors
$\kappa_D,\kappa_C\in(0,1]$, diversity smoothing $\delta_D>0$, and the
actual regularized standard deviations $\sqrt{\operatorname{Var}
+\sigma_b^2}$, $\sigma_b>0$.
Write the two positive logistic base floors as $f_b>0$ and their
amplitudes as $A_b>0$, $b=r,s$. Fix $\bar p_r,\bar p_s>0$ and use
$p_b=\theta\bar p_b$. The acceptance scale and denominator
regularizer are $s_c,\epsilon_c>0$. Put
$$
\begin{aligned}
M&=\sum_b\bar p_b\max\{|\log f_b|,|\log(f_b+A_b)|\},\\
\Delta&=\sum_b\bar p_b\log[(f_b+A_b)/f_b],\qquad
D_0=e^{-M}+\epsilon_c,\\
a_0&=\frac{e^M\Delta}{s_cD_0},\qquad
c_0=\frac{a_0}{\kappa_C},\qquad
G_0=\frac1{s_cD_0}+\frac{e^M\Delta}{s_cD_0^2},\\
J_b&=\frac{e^M\bar p_bA_b}{4f_b},\qquad
K_D^f=1+4/\kappa_D,\qquad
S_b=\sqrt{D_*^2+\delta_D^2}-\delta_D,\\
T_r&=R_{\rm osc}/\sigma_r+R_{\rm osc}^3/(2\sigma_r^3),\qquad
T_s^f=S_b/\sigma_s+S_b^3/(2\sigma_s^3),\\
C_{r,0}&=J_rT_r,\qquad C_{s,0}=J_sT_s^f,\\
L_{f,0}&=4c_0+(4c_0+2a_0)K_D^f
                +2G_0(C_{r,0}+K_D^fC_{s,0}),\\
D_{r,0}&=2a_0/\kappa_C^2+2G_0C_{r,0}/\kappa_C,\qquad
D_{m,0}=2G_0C_{s,0}/\kappa_C,\\
D_{G,0}&=2c_0K_D^f+8c_0/\kappa_D
                    +2D_{r,0}+2D_{m,0}K_D^f,\\
S_{L,0}&=2a_0+4c_0,\qquad S_{G,0}=L_{f,0}+2D_{G,0},\\
K_{f,0}&=1+3L_{f,0},\qquad J_{\rm conn,0}=30c_0,\\
P_{L,0}&=4c_0(1+S_{L,0})+2c_0,\\
P_{G,0}&=4c_0[S_{G,0}+(1+2J_{\rm conn,0})K_{f,0}]+D_{G,0},\\
A_{G,0}&=S_{G,0}+J_{\rm conn,0}K_{f,0}+P_{G,0},\\
H_0&=24c_0+9L_{f,0},\qquad J_0=3c_0(20+42L_{f,0}),\\
C_{\rm comb,0}
&=\max\{H_0+2d\sigma_J^2[\eta(P_{L,0}+P_{G,0})+gJ_0],\\
&\hspace{22mm} S_{L,0}+P_{L,0},\
 H_0+J_0+(\eta/g)A_{G,0}\}.
\end{aligned}
$$
For every $0<\theta\le\min\{1,\kappa_C/(8a_0)\}$, the actual
complete preparation coupling satisfies
$$
\mathbb E D_{\eta,g}(S^{\rm prep},T^{\rm prep})
\le[1+\theta C_{\rm comb,0}]D_{\eta,g}(S,T).
$$
All constants are finite, population independent, and determined
by the displayed primitive parameters and the kinetic weights.
:::

:::{prf:proof}
The actual positive-base fitness obeys $e^{-M}\le F\le e^M$
and $F^*-F_*\le\theta e^M\Delta$ for $0<\theta\le1$.
Thus its acceptance ceiling satisfies $a_*\le\theta a_0$,
its per-label accepted mass constant satisfies $c_*\le\theta c_0$,
and its gate derivative constant satisfies $L_g\le G_0$.
For the actual logistic-power derivative, divide the full fitness
by its positive base and differentiate that base. This gives
$H_b\le\theta J_b$.
The finite empirical mean and variance comparison for two bounded
arrays therefore gives
$C_r^f\le\theta C_{r,0}$ and
$C_s^f\le\theta C_{s,0}$, retaining their computed global
normalizers.

Import the complete finite-preparation coupling of research record 17,
equations (KUWP.13)--(KUWP.23). Its actual factor is
$$
C_{\rm comb}=
\max\{C_H+2d\sigma_J^2(\eta P_J+gC_J),\
          A_L,\ C_E+(\eta/g)A_G\}.
$$
The preceding bounds give $L_f\le\theta L_{f,0}$ and
$c_*\le1/8$. Hence its conditional-forest size bound
$M_f=e^{8c_*}$ obeys $M_f<3$ and
$M_f-1\le24\theta c_0$.
Its specified-vertex connection bound obeys
$$
J_{\rm conn}=(8c_*+16c_*^2)e^{8c_*}
\le30\theta c_0.
$$
Substitution into the complete preparation constants yields
$$
\begin{gathered}
C_H-1\le\theta H_0,\qquad C_J\le\theta J_0,\qquad
C_E-1\le\theta(H_0+J_0),\\
P_L\le\theta P_{L,0},\quad P_G\le\theta P_{G,0},\quad
A_L-1\le\theta(S_{L,0}+P_{L,0}),\quad
A_G\le\theta A_{G,0}.
\end{gathered}
$$
Here products of two $\theta$ factors were bounded by one
$\theta$ using $\theta\le1$; their other factors remain in the
displayed constants. The three alternatives in $C_{\rm comb}$
are therefore at most $1+\theta C_{\rm comb,0}$.
This is the actual forest and source-energy estimate, including
exceptional donor endpoints and component-Haar collisions, rather
than an assumed preparation continuity modulus. $\square$
:::

:::{prf:theorem} Complete active conservative law at a nonresonant harmonic step
:label: thm-kuhw-active-exact-uniform-law

Retain the actual bounded-reward preparation and the nonviscous
harmonic kinetic register above. Assume all comparison-feature maps
are continuous, death and history terms are disabled, and the actual
component collision contracts the original frozen-slot velocity energy, as the canonical
orthogonal mean/relative-velocity collision does for
$|\alpha_{\rm col}|\le1$. Use
$V_*=(1+2|\alpha_{\rm col}|)V$ in the kinetic register.
Define
$$
\theta_{\rm hw}=
\min\left\{
1,\ \frac{\kappa_C}{8a_0},\
\frac{1-\lambda}{2\lambda c_0},\
\frac{1-q_D}{2q_DC_{\rm comb,0}}
\right\}>0.
$$
For every fixed $0<\theta\le\theta_{\rm hw}$, the actual complete
finite swarm kernel $P_{N,\theta}$ satisfies, for every $N\ge2$,
$$
\mathbb E D_{\eta,g}(S^+,T^+)
\le q_*D_{\eta,g}(S,T),\qquad q_*=\frac{1+q_D}{2}<1.
$$
Its own averaged moment obeys
$$
P_{N,\theta}\mathsf M_2(S)\le
\lambda_*\mathsf M_2(S)+B_*,
\qquad
\lambda_*=\frac{1+\lambda}{2}<1,\qquad
B_*=B_2+\lambda d\sigma_J^2a_0.
$$
It has a unique invariant law $\Pi_{N,\theta}$ in the
finite-$\mathsf M_2$ class, with
$$
\int W(z_i)\,\Pi_{N,\theta}(dS)
\le H_*:=\frac{B_*}{1-\lambda_*}
\qquad\text{for every }i,N.
$$

For any entering law $\mu_N$ of finite averaged moment, put
$$
A_{\mu_N}^*=1+(\eta+g)
 \left[\int\mathsf M_2(S)\,\mu_N(dS)+H_*\right].
$$
Let $\mathscr L_{N,n}$ be the law of the random all-alive empirical
measure under $\mu_NP_{N,\theta}^n$, let
$\mathscr L_{N,\infty}=(\mathcal E_N)_\#\Pi_{N,\theta}$, and
let $\lambda_{N,n}^a,\lambda_{N,\infty}^a$ be their respective
mean empirical measures. For every positive physical matrix $G$,
$$
\begin{aligned}
\|\lambda_{N,n}^a-\lambda_{N,\infty}^a\|_{\rm TV}
&\le A_{\mu_N}^*q_*^n,\\
\mathcal W_{2,N,G}^2(\mu_NP_{N,\theta}^n,\Pi_{N,\theta})
&\le\frac{2\lambda_{\max}(G)}{\eta}A_{\mu_N}^*q_*^n,\\
\mathcal W_{2,\mathrm{emp},G}^2
  (\mathscr L_{N,n},\mathscr L_{N,\infty})
&\le\frac{2\lambda_{\max}(G)}{\eta}A_{\mu_N}^*q_*^n,\\
W_{2,G}^2(\lambda_{N,n}^a,\lambda_{N,\infty}^a)
&\le\frac{2\lambda_{\max}(G)}{\eta}A_{\mu_N}^*q_*^n.
\end{aligned}
$$
These are exact finite-population law targets with no sampling floor.
The rate and moment constants are independent of $N$.
The raw reward is the configured bounded channel in the theorem;
an unbounded same-potential reward is not silently replaced by it.
No near-equal-fitness region is excluded.
:::

:::{prf:proof}
The exponent interval makes $a_*\le1/8$ and supplies the complete
actual preparation estimate. The kinetic coupling applies
conditionally to its full prepared arrays, including their
component correlations. The tower property gives a composed
coefficient at most
$$
q_D(1+\theta C_{\rm comb,0})\le(1+q_D)/2=q_*<1.
$$
This constructs a measurable coupling of the actual complete
one-update marginals.

For the own moment, an accepted token has probability
$b_{ij}\le a_*/[\kappa_C(N-1)]$. Each donor column therefore has
incoming mass at most $c_*$. Summing the exact copied/retained
position mixtures gives a pre-jitter averaged positional moment
at most $(1+c_*)\mathsf M_{2,x}(S)$.
The centered activated jitters add at most
$d\sigma_J^2a_*$ in expectation. The actual component collision
uses the original frozen slot velocities and contracts their
total velocity energy. Thus the phase moment is bounded by
$(1+c_*)\mathsf M_{2,x}(S)+\mathsf M_{2,v}(S)$ before
the jitter contribution, which is at most
$(1+c_*)\mathsf M_2(S)$. Consequently
$$
\mathbb E\mathsf M_2(S^{\rm prep})
\le(1+c_*)\mathsf M_2(S)+d\sigma_J^2a_*.
$$
Apply the kinetic drift after that actual preparation. The exponent
interval implies $\lambda(1+c_*)\le(1+\lambda)/2=\lambda_*$,
while $a_*\le\theta a_0\le a_0$, proving the displayed full drift.

For fixed $N$ the complete kernel is Feller on the closed
finite-dimensional state space with capped velocities.
Indeed the finitely many measurement and donor choices have
continuous positive-normalizer probabilities; the actual
regularized means, variances, logistic fitnesses and clipped
gates are continuous. For each fixed accepted graph the
component collision is continuous in the frozen physical
velocities, and the integrals over its prescribed Haar
rotations and Gaussian jitters are continuous on bounded
continuous tests by dominated convergence. Summing those
finite graph contributions and composing with the continuous
harmonic kinetic/cap update proves the claim. A fitness tie
causes no discontinuity: the probability of the affected
accepted edge tends to zero.

Starting at the zero array, the own drift gives
$\sup_n\mathbb E\mathsf M_2(S_n)\le H_*$.
Thus its empirical time averages are tight for this fixed $N$:
the total second moment of all coordinates is at most $NH_*$,
and velocities stay capped. Extract a weakly convergent
subsequence of those averages. For every bounded continuous
$\varphi$, their invariance residual is
$$
\frac1T\sum_{n=0}^{T-1}
\mathbb E[P_{N,\theta}\varphi(S_n)-\varphi(S_n)]
=\frac{\mathbb E\varphi(S_T)-\varphi(0)}T\longrightarrow0.
$$
The Feller property passes this identity to the weak limit,
which is therefore invariant. Lower semicontinuity of
$\mathsf M_2$ gives its finite moment at most $H_*$.
This uses the confining moment drift on the full unbounded
space, not a position cutoff.

For uniqueness, couple any two finite-moment invariant laws
initially, and iterate the constructed complete pair coupling.
The initial expected cost is bounded by
$1+(\eta+g)[\mu\mathsf M_2+\mu'\mathsf M_2]<\infty$.
At time $n$ the expected mismatch fraction is at most that
initial cost times $q_*^n$. Full-array TV is at most the
probability of any mismatched label, at most $N$ times that
fraction. For this fixed $N$ it tends to zero while both
marginals remain invariant, proving uniqueness.
No triangle inequality or Banach theorem for $D_{\eta,g}$ is used.
Permutation equivariance of the actual kernel and uniqueness
make its invariant law exchangeable, converting its averaged
moment bound to the stated individual bound.

Finally couple $\mu_N$ to this invariant law initially, for
example independently. Its expected cost is at most
$A_{\mu_N}^*$, and the actual pair iteration bounds that cost
at time $n$ by $A_{\mu_N}^*q_*^n$.
The physical cost inequality in the preceding corollary
gives the full-swarm Wasserstein bound.
Pushing the same coupling through empirical matching and
sampling one common independent uniform label give the
remaining three alive-law estimates with their exact
stationary marginals. Every primitive denominator is
positive and finite, and both $\lambda,q_D$ have strict
gaps, so $\theta_{\rm hw}>0$. $\square$
:::

:::{prf:corollary} A fully specified active profile at the original step size
:label: cor-kuhw-original-step-active-profile

In dimension $d=3$, fix
$$
h=1/25,\quad\gamma=b_O=1,\quad\sigma_x=1/10,\quad
V=2,\quad\alpha_{\rm col}=1/2,\quad\sigma_J=1/10,\quad\nu=0,
$$
and the actual configured channels
$$
F(x)=-x,\qquad R(x)=-\tanh(|x|^2/2).
$$
Use continuous comparison features with position and velocity radii
$R_x^{\rm feat}=R_v^{\rm feat}=1$, relative-velocity coefficient one,
role widths $\epsilon_D=\epsilon_C=4$, and
$\delta_D=1/1000$. Set both logistic floors and amplitudes to one,
both standardization regularizers and both acceptance parameters
$s_c,\epsilon_c$ to one, and $\bar p_r=\bar p_s=1$.
Take $r=u=R_{\rm ref}=1$ in the kinetic register. Then
$D_*\le\sqrt8$; use the certified values
$D_*=\sqrt8$, $\kappa_D=\kappa_C=e^{-1/4}$ and $R_{\rm osc}=1$.
Compute every constant above from these values and choose
$$
p_r=p_s=\theta_{\rm hw}/2>0.
$$
The complete actual conservative law satisfies all the
population-uniform bounds in the preceding theorem.
:::

:::{prf:proof}
The original step gives $t=1/50$, $0<c=e^{-1/25}<1$,
$m=1-t^2>0$, $0<a_x=1-t^2(1+c)<1$,
$q^2=(1-e^{-2/25})/2>0$ and $s^2=1/2500>0$.
The collision envelope is $V_*=4$ and its orthogonal
mean/relative-velocity law contracts original frozen-slot velocity energy.
The configured reward is continuous with range contained
in $[-1,0]$, the comparison features have the stated
finite diameter, and all floors, widths and regularizers
are strictly positive. Thus every primitive constant is
finite and $\theta_{\rm hw}>0$, so the chosen powers
lie in the proved interval.
The channels are explicitly those displayed; no
bounded force-center resonance is assumed.
Cloning is active on unequal-fitness inputs: already
two zero-velocity walkers at $0$ and $e_1$ have unequal
reward factors, identical sampled diversity factors,
and a strictly positive inferior-to-superior
acceptance gate for these positive powers. Complete
fitness ties remain covered and have zero acceptance.
$\square$
:::

:::{prf:theorem} Frozen population preparation feedback with a common root
:label: thm-kuhw-frozen-provider-fourth-feedback

Let $J_\mu(z,\cdot)$ be the actual population rooted-component
preparation provider: its physical root is fixed at $z$, its
measurement companion is drawn from the actual normalized
$P_D(\cdot\mid z,\mu)$, and its fitness uses the actual reward and
sampled-diversity means and regularized variances in environment $\mu$.
Its outgoing accepted token and incoming marked Poisson children
generate the complete ordered rooted component. Copy the selected
frozen source position, retain the original frozen-slot velocities
as inputs to its actual component-Haar collision, and activate
the recipient Gaussian jitter exactly on acceptance. No kinetic
update is included in $J_\mu$.

Use bounded continuous raw reward and the positive logistic
primitive register above. This provider is the population tree,
rather than a finite array with its random empirical standardizers.
Suppose $\mu,\mu'$ are supported on velocities capped at $V$ and
have eighth positional moments at most $H_8<\infty$. Define
$$
w_4(z)=1+|x|^4,\qquad
\|\xi\|_{W_4}=\int w_4\,d|\xi|,\qquad
\delta_4(\mu,\mu')=\tfrac12\|\mu-\mu'\|_{W_4},
$$
$$
B_w=1+\sqrt{H_8},\qquad
K_D=1+2/\kappa_D,\qquad K_{D,w}=1+2B_w/\kappa_D,
$$
$$
\begin{aligned}
C_{F,0}&=J_rT_r+J_sK_DT_s^f,\\
D_{\beta,0}&=a_0/\kappa_C^2+2G_0C_{F,0}/\kappa_C,\\
L_0^{\rm env}&=2c_0K_D+D_{\beta,0},\\
D_J&=1+(d+2)\sigma_J^2+d(d+2)\sigma_J^4,\\
C_J^{\rm env}&=D_J\left[
\frac{4(a_0+2c_0+c_0B_w)}{\kappa_D}
+4(c_0K_{D,w}+D_{\beta,0}B_w+L_0^{\rm env})
+24L_0^{\rm env}(1+c_0B_w)\right],\\
L_J&=B_wC_J^{\rm env}.
\end{aligned}
$$
Then for every
$0<\theta\le\min\{1,\kappa_C/(8a_0)\}$ and every fixed physical root,
$$
\boxed{\quad
\|J_\mu(z,\cdot)-J_{\mu'}(z,\cdot)\|_{W_4}
\le\theta C_J^{\rm env}w_4(z)\,
                      \|\mu-\mu'\|_{W_4}.
\quad}
$$
Consequently, for every common root law $\zeta$ with
$\zeta|x|^4\le\sqrt{H_8}$,
$$
\boxed{\quad
\|\zeta J_\mu-\zeta J_{\mu'}\|_{W_4}
\le\theta L_J\|\mu-\mu'\|_{W_4}.
\quad}
$$
The comparison has an $O(\theta)$ coefficient because the physical
root is common. It is not the complete-law comparison
$\mu J_\mu-\mu'J_{\mu'}$, whose entering physical laws already differ.
The bound retains the actual population sampled normalizers and the
complete component tree.
:::

:::{prf:proof}
Write $\delta=\delta_4(\mu,\mu')$. Each environment has mean
$w_4$ at most $B_w$ by Cauchy--Schwarz.
Couple ordinary marked population types by first maximally
coupling their physical laws, then maximally coupling the
measurement companions at matching physical roots.
Subtracting the normalized companion numerators gives
conditional measurement failure probability at most
$2\delta/\kappa_D$. Thus the ordinary type coupling has
bad probability at most $K_D\delta$ and weighted bad mass
at most
$$
\int(w_4(z_1)+w_4(z_2))\mathbf1_{\rm bad}\,d\Lambda
\le2\delta+4B_w\delta/\kappa_D=2K_{D,w}\delta.
$$
The physical unmatched weighted mass is exactly $2\delta$.
The measurement-failure probability is uniform over matched
physical states, which justifies its weighted integration.

The reward mean and variance comparison for a bounded scalar
range gives $T_r\delta$. Apply the same comparison to the
coupled sampled-diversity law, whose mismatch probability
is at most $K_D\delta$, to get $K_DT_s^f\delta$.
The actual logistic-power derivative bounds therefore give
$$
|F_\mu-F_{\mu'}|
\le\theta C_{F,0}\delta
$$
on matching underlying physical/measurement types, including
the difference between their numeric global normalizers.
The gate bounds satisfy $a_*\le\theta a_0$,
$c_*\le\theta c_0$ and $L_g\le G_0$.
For good recipient and donor types, subtracting the actual
donor normalizers and gates bounds the accepted-edge density
difference by $D_\beta\delta$, where
$$
D_\beta\le\theta D_{\beta,0},\qquad
L:=2c_*K_D+D_\beta\le\theta L_0^{\rm env}.
$$
Bad paired types cost at most $2c_*K_D\delta$ in unweighted
accepted mass. Consequently an outgoing accepted
subprobability, completed by its no-edge outcome, and
the incoming Poisson intensities can be coupled with
total mismatch hazard at most $2L\delta$ per queried
matched vertex.

The actual Gaussian jitter readout has a uniform source-weight
bound independent of the collision graph. If the root's frozen
source position is $y$ and its acceptance indicator is $I$,
its prepared position is $y+I\sigma_JZ$, and
$$
\mathbb E|y+\sigma_JZ|^4
=|y|^4+(2d+4)\sigma_J^2|y|^2+d(d+2)\sigma_J^4.
$$
Use $2|y|^2\le1+|y|^4$. Whether the jitter is used or not,
$$
\mathbb Ew_4(y+I\sigma_JZ)\le D_J(1+|y|^4).
$$
The collision affects velocity only, so this bound applies
to every prescribed component-Haar realization. It does
not assert independence of that component and its frozen
source. For a test $|\varphi|\le w_4/2$, every conditional
preparation readout therefore has absolute value at most
$D_Jw_4(y)/2$.

Fix the same physical root $z$ and initially couple its two
measurement marks. First suppose those marks match.
Expose its outgoing token before exploring any other edge.
The source position is thereby identified: it is the root
if that token is no edge and its donor if it is accepted.
Incoming cloners and the remaining component do not change
this frozen source. A failed outgoing coupling has
weighted test cost at most
$$
4D_J(c_*K_{D,w}+D_\beta B_w+Lw_4(z))\delta.
$$
Indeed, bad donor types contribute their weighted bad mass
times $c_*$, good donor density differences contribute at
most $2D_\beta B_w\delta$, and the no-edge completion
contributes at most $2Lw_4(z)\delta$; the displayed factor
four is a conservative bound after the readout.

If the outgoing token and its types match, expose that
result and then sample remaining graph primitives freshly
with their actual conditional marginal laws.
The first-marginal full component after one exposed edge
has conditional mean size at most $2M_{\rm graph}$,
$M_{\rm graph}=e^{2c_*}$. To verify this uniform bound,
every simple path in the accepted outdegree-one forest
has an increasing-fitness arm followed by a decreasing
arm. With total length $\ell$ and arm lengths $k,\ell-k$,
the independent marked integrations have ordered volumes
at most $1/[k!(\ell-k)!]$ and edge densities at most
$c_*^\ell$. Summing arm lengths and then $\ell$ gives
$e^{2c_*}$. Strict ordering also covers atomic marked
laws because ties forbid accepted edges. Deleting the
exposed edge leaves at most two free-root components.
A consumed outgoing edge and a known incoming child
are excluded by vertex identity; the other incoming
points have their original Palm Poisson law.

Complete that first marginal without conditioning its
unexposed primitives on future coupling successes.
The queried matched vertices up to the first discrepancy
are a pathwise subset of this completed component.
Summing their conditional hazards bounds remaining
component failure probability by
$4LM_{\rm graph}\delta$, uniformly in the exposed
source type. Its common frozen source has mean weight
at most $w_4(z)+c_*B_w$. Hence the matched-source
component test cost is at most
$$
4D_JL(1+c_*B_w)M_{\rm graph}w_4(z)\delta.
$$
This uses a conditional bound uniform in the source;
it does not multiply two dependent unconditional
probabilities or a component weight and its count.

Finally consider a failure of the root-mark coupling,
whose probability is at most $2\delta/\kappa_D$.
Conditional on each own mark, draw its component using
fresh primitives with its prescribed marginal law.
Subtract the common identity-preparation baseline
$\varphi(z)$ before comparing the two readouts.
An isolated root has zero increment. In one environment
the chance of an outgoing root edge is at most $a_*$
and that of an incoming edge is at most $c_*$.
The expected frozen-source weight on outgoing edges
is at most $c_*B_w$. The absolute expected increment,
conditional on any root mark, is therefore at most
$$
\frac{D_J}{2}[(a_*+2c_*)w_4(z)+c_*B_w].
$$
Summing the two own increments and the mark-failure
probability costs at most
$2D_J(a_*+2c_*+c_*B_w)w_4(z)\delta/\kappa_D$.
The mark-coupling failure coin does not condition either
own component further than its own mark.

The preceding three contributions are bounded, with
spare factors two, by
$$
D_J\left[
\frac{4(a_*+2c_*+c_*B_w)}{\kappa_D}
+4(c_*K_{D,w}+D_\beta B_w+L)
+8L(1+c_*B_w)M_{\rm graph}
\right]w_4(z)\delta.
$$
Because $c_*\le1/8$, $M_{\rm graph}<3$.
Insert the linear envelopes and $\theta\le1$ to bound
this bracket by $\theta C_J^{\rm env}w_4(z)\delta$.
Taking the supremum over $|\varphi|\le w_4/2$ proves
the half-normalized weighted-variation bound.
Multiplying both sides by two gives the displayed
full-variation version with the same coefficient.
Integrate over the common root law and use
$\zeta w_4\le B_w$ to obtain the provider bound.
Every graph, source, Haar and jitter primitive kept
its exact own marginal throughout. $\square$
:::

:::{prf:remark} The remaining dense-viscosity inference
:label: rem-kuhw-count-extension

For $\nu>0$, the actual first count kick depends on all entering rows
and the second kick depends on all noisy positions. Even an equal
entering row need not have an equal one-row conditional kernel in
the two populations. The independent row coupling above therefore
does not describe the actual dense kernel.

The positive count-viscosity physical quadratic contraction proved
in research record 15 remains valid, but it does not imply an exact
weighted-Hamming reset. To retain this combined-cost argument one
must construct a coupling of the full correlated count kernel and
prove a bound of the form
$$
\mathbb E D_{\eta,g}(S_\nu^+,T_\nu^+)
\le(q_D+\nu C_{\rm count})D_{\eta,g}(S,T)
$$
with a finite primitive $C_{\rm count}$ independent of $N$, or prove
a compatible full-path block bound. Comparing each viscous law with
its product reference gives an additive discrepancy and does not
prove this pair-dependent estimate. A global Gaussian density
comparison alone can also introduce a population factor. Neither
comparison is used as an established count-kernel coupling here.
:::
