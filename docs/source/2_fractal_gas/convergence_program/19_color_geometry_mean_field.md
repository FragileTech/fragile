(sec-cg-mean-field)=
# The population limit of the recorded viscous gas

(sec-cg-mf-map)=
## 1. The exact population maps for the two existing normalizations

:::{div} feynman-prose
Follow a walker through one update and ask what the rest of the swarm tells
it. At the first force kick, it sees the positions and velocities left by
cloning and collision. Then the walkers move, receive their thermostat
innovations, and move again. By the second kick the population has changed.
The second force must use that changed population. Reusing the earlier
force would describe a different machine.

The population limit keeps this order intact. Replace the finite average
in a kick by an integral against the law at that very stage. Everything
else stays where it was: collision, the two drifts, the two noises, the
velocity cap, and the terminal alive/dead mark. Time still advances by the
configured update duration. The positive convergence result below applies
to the existing quadratic reference gas, with its color and geometry
observations retained. The parameter register makes each estimate
reproducible and lets us see which parameters the kinetic calculation
actually uses. Selection continues to carry its own previously proved
requirements.
:::

:::{prf:definition} Full parameter register for the population estimates
:label: def-cg-mf-parameter-register

The algorithm data are the complete canonical tuple $\theta$ of
{prf:ref}`def-slc-parameter-register`, together with

$$
\Theta=(\theta;\nu,\rho,\mathsf n,\mathsf G_{\vartheta_G},
\mathsf R_{\vartheta_R},c_{\rm phys},\hbar_{\rm eff};\ell_*,t_*).
$$

Here $\mathsf n\in\{\mathrm{count},\mathrm{row}\}$ is the existing force
normalization bit, fixed throughout an update and its population limit.
The force-moment and conditional-concentration estimates explicitly marked
count normalization retain that scope. $\mathsf G_{\vartheta_G}$ and $\mathsf R_{\vartheta_R}$ are the
declared graph/geometry and physical observable readouts of
{prf:ref}`def-variant-recorded-color-geometry`, with all their measurement
parameters retained. $c_{\rm phys}>0$ and $\hbar_{\rm eff}>0$ are the
declared speed and action calibrations; neither changes this transition.
There is no donor history and no geometry feedback. The force and reward
are separate specified landscape functions $F=-\nabla U$ and $R$.
For the transport norms and scalar error budgets choose and record
reference length $\ell_*>0$ and time $t_*>0$. All formulas in this
chapter use their nondimensional coordinates: $x=x_{\rm phys}/\ell_*$,
$v=t_*v_{\rm phys}/\ell_*$, $h=h_{\rm phys}/t_*$,
$\rho=\rho_{\rm phys}/\ell_*$, $\nu=t_*\nu_{\rm phys}$,
$F=t_*^2F_{\rm phys}/\ell_*$, $L_F=t_*^2L_{F,\rm phys}$,
$q=t_*q_{\rm phys}/\ell_*$, $s=s_{\rm phys}/\ell_*$ and
$V=t_*V_{\rm phys}/\ell_*$, with
$\gamma=t_*\gamma_{\rm phys}$ and
$b_O=t_*^{3/2}b_{O,\rm phys}/\ell_*$, with
$\sigma_x=\sqrt{t_*}\sigma_{x,\rm phys}/\ell_*$ and
$\sigma_J=\sigma_{J,\rm phys}/\ell_*$. Clone jitter, position feature radii
and donor radii scale as positions; velocity feature radii scale as
velocities. Explicitly, $R_x^{\rm feat}=R_{x,\rm phys}^{\rm feat}/\ell_*$,
$R_v^{\rm feat}=t_*R_{v,\rm phys}^{\rm feat}/\ell_*$,
$\lambda_{\rm alg}=\lambda_{{\rm alg},\rm phys}/t_*^2$,
$\epsilon_b=\epsilon_{b,\rm phys}/\ell_*$ for $b=D,C$,
$\delta_D=\delta_{D,\rm phys}/\ell_*$ and
$\sigma_s=\sigma_{s,\rm phys}/\ell_*$. Thus companion distances
and widths have the same length conversion and their probability ratios
are unchanged; diversity values and their standardizer scale together.
Reward values and their standardizer keep their configured reward unit,
with $R(x,v)=R_{\rm phys}(\ell_*x,\ell_*v/t_*)$.
Fitness-map amplitudes, floors, exponents, acceptance data and collision
restitution are unchanged dimensionless entries. Geometry and
color are evaluated in their original recorded units after the inverse
coordinate conversion; in particular
$\kappa=\kappa_{\rm phys}\ell_*/t_*$ in a nondimensional color phase.
This declaration makes the sums of position and velocity
costs below dimensionally defined; it changes units, not the algorithm.
Compute the following profiles of the configured force and domain:

$$
L_F=\sup_{x\ne y}\frac{|F(x)-F(y)|}{|x-y|},
\qquad f_0=|F(0)|,
\qquad B_D=\sup_{x\in D}|x|<\infty
$$

for the terminal-domain corollary. These are profiles, not new assumptions
on a landscape: a divergent profile makes its displayed bound infinite.
The positive population-limit conclusions below concern the unchanged
quadratic reference instance of
{prf:ref}`rem-variant-viscous-euclidean-rust`, for which
$U(x)=\lambda|x|^2/2$ gives $L_F=\lambda$, $f_0=0$ and the configured
terminal box gives $B_D=\sqrt d L_D$. At its numerical preset,
$\lambda=1$, $d=3$, $L_D=2$, hence $L_F=1$ and $B_D=2\sqrt3$ in the
preset units. Bounds for other forces are evaluated from their profiles,
without replacing the configured force. The reward regularity,
standardization, donor weights and acceptance parameters needed by
the selection theorem retain exactly its displayed hypotheses;
they are not inferred from $L_F$. The noise parameters are

$$
c_h=e^{-\gamma h},\quad
q=b_O\sqrt{\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,
\end{cases}}
\quad s=\sigma_x\sqrt h,\quad \sigma_J=\sigma_{\rm clone},\quad
R_c=(1+2|\alpha_{\rm col}|)V.
$$

Every bound below is a function of this register and of the explicitly
named test/moment order and input law. A constant that uses only
$(d,h,\nu,\rho,L_F,f_0,q,s,V,R_c)$ is independent of the other
entries because those entries are not consumed by the kinetic stages;
the preceding cloning theorem still retains their effect.
:::

:::{prf:definition} Coupled kinetic map at fixed algorithmic time
:label: def-cg-mf-kinetic-map

Use either existing Gaussian-kernel normalization of the Viscous Euclidean Gas of
{prf:ref}`def-variant-viscous-euclidean`, with $h>0$, $\nu\ge0$,
$\rho>0$, globally Lipschitz force $F:\mathbb R^d\to\mathbb R^d$,
thermostat coefficient $c_h\in[0,1]$, OU amplitude $q>0$, final position
amplitude $s=\sigma_x\sqrt h>0$, and cap radius $V>0$. Put $a=h/2$ and

$$
K_\rho(x,y)=e^{-|x-y|^2/(2\rho^2)},\qquad
C_\lambda(x,v)=\int K_\rho(x,y)(w-v)\,\lambda(dy,dw),\qquad
a_\lambda(x)=\int K_\rho(x,y)\,\lambda(dy,dw),
$$

$$
\mathcal V_\lambda^{\rm count}(x,v)=\nu C_\lambda(x,v),\qquad
\mathcal V_\lambda^{\rm row}(x,v)=\nu C_\lambda(x,v)/a_\lambda(x).
$$

The Gaussian kernel is strictly positive, so $a_\lambda(x)>0$ for every
$x$ and probability law $\lambda$. Finite first velocity moments make the
numerator finite. The unqualified symbol $\mathcal V_\lambda$ in the
count-normalized force estimates below means
$\mathcal V_\lambda^{\rm count}$. The selected bit specifies
$\mathcal V_\lambda^{\mathsf n}$ in both kicks of the map.

Starting from the actual post-collision law $\lambda_0$, define

$$
\begin{aligned}
v_1&=v+a[F(x)+\mathcal V_{\lambda_0}^{\mathsf n}(x,v)],
&x_1&=x+av_1,\\
v_2&=c_hv_1+q\xi,
&x_2&=x_1+av_2,
&\lambda_2&=\operatorname{Law}(x_2,v_2),\\
v_3&=v_2+a[F(x_2)+\mathcal V_{\lambda_2}^{\mathsf n}(x_2,v_2)],
&x^+&=x_2+s\zeta,\\
v^+&=\psi_V(v_3),
&b^+&=\mathbf1_D(x^+).
\end{aligned}
$$

Here $\xi,\zeta$ are independent standard $d$-dimensional Gaussians.
For an unbounded conservative instance take $D=\mathbb R^d$ and $b^+=1$.
Denote this output law by $\mathcal K_h^\nu(\lambda_0)$. The complete
population update is $\mathcal F_h^\nu=\mathcal K_h^\nu\circ\mathcal J$,
where $\mathcal J$ is the component collision/selection law of
{prf:ref}`thm-mean-field-one-step-consistency`. An ordered donor-star
implementation uses its own priority-marked $\mathcal J$, as specified in
{prf:ref}`def-chaos-ordered-star-population-map`.

At finite $N$, replace $\lambda_0,\lambda_2$ by the empirical measures
at those same two stages. For count normalization this is exactly the
finite force: the self-term vanishes since $w-v=0$. For row normalization
at an actual empirical row $(x_i,v_i)$ and $N\ge2$, the exact finite force is

$$
F_i^{\mathrm{visc},N,\rm row}
=\nu\frac{C_{L_N}(x_i,v_i)}{a_{L_N}(x_i)-1/N}.
$$

The subtraction removes the self-weight $K_\rho(x_i,x_i)/N=1/N$
from the denominator; the self-term in the numerator is already zero.
For $N=1$ the configured force is zero. The two force laws refer to
different stages and cannot be replaced by the input law at both kicks.
All parameters stay fixed during the population limit. No continuous-time
generator, new spatial cutoff, or change to the cloning rule is introduced.
For physical color and spatial geometry use the recorded variant of
{prf:ref}`def-variant-recorded-color-geometry` with $d=3$.
:::

(sec-cg-mf-stability)=
## 2. Force stability and the conditional noise estimate

:::{div} feynman-prose
Here is the calculation that lets the population argument survive a viscous
force. Match walkers in two populations. A difference in the force can
come from their velocities, or from their positions changing the weights
assigned to those velocities. The Gaussian weight has a bounded spatial
slope, so we can estimate both effects. Large velocities still matter:
the thermostat has not been capped when the second force is evaluated.

That is why the moment calculation follows every intermediate stage.
The final cap bounds the completed velocity, but it cannot supply a bound
for a velocity used earlier. The count normalization gives a useful
alternative to following the fastest walker: bound the population average
of the velocity moments. Those estimates remain independent of the number
of rows. Fresh Gaussian draws provide a separate concentration estimate
when we condition on the entire array entering that noise stage. After a
coupled kick, the rows can be dependent again; the argument keeps that
dependence instead of discarding it.
:::

:::{prf:lemma} Stability of the empirical viscous force
:label: lem-cg-viscous-force-stability

The Gaussian kernel satisfies $0<K_\rho\le1$ and is Lipschitz in each
argument with constant $L_\rho=e^{-1/2}/\rho$. Let $\pi$ couple
$\lambda,\eta$ and put

$$
D_p=\left[\int(|x-x'|^2+|v-v'|^2)^{p/2}\,d\pi\right]^{1/p},
\qquad M_4(\eta)=\left(\int|v'|^4\,d\eta\right)^{1/4}.
$$

For finite fourth moments,

$$
\left\|\mathcal V_\lambda(x,v)
       -\mathcal V_\eta(x',v')\right\|_{L^2(\pi)}
\le \nu[2+4L_\rho M_4(\eta)]D_4.
$$

If both velocity marginals are supported in $\overline B(0,R)$, then
for every $p\ge1$,

$$
\left\|\mathcal V_\lambda(x,v)
       -\mathcal V_\eta(x',v')\right\|_{L^p(\pi)}
\le\nu(2+4L_\rho R)D_p.
$$

These statements apply to empirical measures with their actual dependence;
they do not require independent input rows.
:::

:::{prf:proof}
The maximum of $r\rho^{-2}e^{-r^2/(2\rho^2)}$ occurs at $r=\rho$,
giving $L_\rho$. Integrate against an independent copy
$(y,w,y',w')$ of the same coupling. Add and subtract
$K_\rho(x,y)(w'-v')$. The absolute force difference is at most

$$
\nu\left[|v-v'|+\mathbb E|w-w'|
+L_\rho\left\{
 |x-x'|\mathbb E|w'|
 +\mathbb E[|y-y'||w'|]
 +|v'|\bigl(|x-x'|+\mathbb E|y-y'|\bigr)
\right\}\right].
$$

The first two terms have $L^2$ norm at most $2D_4$. Cauchy--Schwarz
controls the first two terms in braces by $2M_4D_4$ in total.
Hölder gives $\||v'||x-x'|\|_2\le M_4D_4$, and the last term
is at most $M_4D_4$. This proves the first inequality. With
$|w'|,|v'|\le R$, Minkowski and Jensen give the stated $L^p$
bound directly. No factorization of $\lambda$ is used.
:::

:::{prf:lemma} Linear-growth moment bounds for both coupled kicks
:label: lem-cg-mf-moments

Let $p\ge1$ and $\|Z\|_{p,N}=(N^{-1}\sum_i|Z_i|^p)^{1/p}$.
For count normalization,

$$
\|F^{\mathrm{visc}}(x,v)\|_{p,N}\le2\nu\|v\|_{p,N}.
$$

The same inequality holds for $\mathcal V_\lambda$ in $L^p(\lambda)$.
Writing $L_F=\operatorname{Lip}(F)$ and $f_0=|F(0)|$, a force kick
therefore satisfies

$$
\|v^{B}\|_{p,N}
\le(1+h\nu)\|v\|_{p,N}+aL_F\|x\|_{p,N}+af_0.
$$

The deterministic drift satisfies
$\|x^A\|_{p,N}\le\|x\|_{p,N}+a\|v\|_{p,N}$.
After expectations, the OU stage has $L^p$ bound
$c_h(\mathbb E\|v_1\|_{p,N}^p)^{1/p}+q(\mathbb E|\xi|^p)^{1/p}$,
and final position noise adds $s(\mathbb E|\zeta|^p)^{1/p}$.
The cap reduces each velocity norm. Consequently the entire kinetic
map preserves finite $p$th moments, with constants independent of $N$.

An explicit moment budget is obtained by the following scalar recursion.
Let $g_{d,p}=[2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2)]^{1/p}$ and let
$X_0,V_0$ bound the $L^p$ mean norms of the post-collision array.
Then use, in the displayed order,

$$
\begin{aligned}
V_1&=(1+h\nu)V_0+aL_FX_0+af_0,&X_1&=X_0+aV_1,\\
V_2&=c_hV_1+qg_{d,p},&X_2&=X_1+aV_2,\\
V_3&=(1+h\nu)V_2+aL_FX_2+af_0,&X_+&=X_2+sg_{d,p},\\
V_+&=\min(V,V_3).
\end{aligned}
$$

For the bounded donor domain take
$X_0=B_D+\sigma_Jg_{d,p}$ and $V_0=R_c$ at every complete update.
Thus $\mathbb E[N^{-1}\sum_i|x_i^+|^p]\le X_+^p$ and
$\mathbb E[N^{-1}\sum_i|v_i^+|^p]\le V_+^p$, explicitly as
functions of the full parameter register. These inequalities do not
replace the unbounded physical output by compact support.

For a bounded terminal donor domain and the unchanged canonical collision
stage, all post-collision velocities are bounded by
$R=(1+2|\alpha_{\rm col}|)V$, and post-collision positions have every
moment bounded uniformly in $N$ and in the incoming nonextinct state:
each position is an alive donor position, possibly plus its prescribed
Gaussian clone jitter. These estimates apply even when retained dead
input positions are unbounded.
:::

:::{prf:proof}
Put $k_{ij}=K_\rho(x_i,x_j)/N$. Both row and column sums of the
nonnegative symmetric matrix $(k_{ij})$ are at most one. Jensen gives

$$
\frac1N\sum_i\left|\sum_jk_{ij}v_j\right|^p
\le\frac1N\sum_{i,j}k_{ij}|v_j|^p
\le\frac1N\sum_j|v_j|^p.
$$

The term $v_i\sum_jk_{ij}$ satisfies the same bound. Minkowski proves
the force estimate. The integral proof replaces both sums by integrals
and uses symmetry of $K_\rho$. Apply the linear-growth bound on $F$
to each B stage, then Minkowski to the drift and Gaussian additions.
The resulting finite sequence of scalar inequalities has no $N$ in its
coefficients. Mandatory revival and the unchanged collision bound in
{prf:ref}`def-slc-parameter-register` prove the terminal-domain assertion.
This is a moment estimate, not long-time attraction on an unbounded domain.
:::

:::{prf:lemma} Empirical convergence through a fresh independent noise stage
:label: lem-cg-mf-noise

Suppose random input arrays $z_i^N$ have empirical laws converging in
probability in $W_4$ to deterministic $\lambda$, and for some $p>4$
satisfy $\sup_N\mathbb E[N^{-1}\sum_i|z_i^N|^p]<\infty$.
Apply to each row the same continuous affine map followed by independent
Gaussian innovations, with a continuous output map of linear growth.
The output empirical laws converge in probability in $W_4$ to the
corresponding pushforward of $\lambda$ and the Gaussian law. For a
bounded test $\phi$, the conditional fluctuation obeys

$$
\operatorname{Var}\left(\frac1N\sum_i\phi(Z_i^N)\ \middle|\ (z_i^N)_i\right)
\le\frac{\|\phi\|_\infty^2}{N}.
$$

This conditional independence statement is invoked only at the fresh
noise stage; it is not asserted after the coupled B2 kick.
:::

:::{prf:proof}
Conditioned on the input array, the innovations are independent.
The conditional empirical mean is the input empirical measure tested
against its Gaussian-averaged bounded continuous function. It converges
by weak convergence of the input. Summing independent centered variances
gives the displayed estimate. Apply this argument to a countable
convergence-determining family of bounded continuous tests. Linear growth
and Gaussian moments give uniform $p$th output moments. For any $R>0$,
the expected empirical fourth moment outside radius $R$ is bounded by
$C R^{4-p}$; Markov's inequality makes this tail small in probability.
Truncation then gives convergence of fourth moments together with weak
convergence, hence $W_4$ convergence. This last assertion follows directly
by coupling the masses in finitely many small cells inside a ball and
charging outside mass by its fourth moment.
:::

(sec-cg-mf-kinetic-limit)=
## 3. The complete coupled kinetic limit

:::{div} feynman-prose
Changing one thermostat draw does more than change one output walker.
That walker moves to a different position and carries a different
velocity into the second kick. Every other row can then feel a change
through the viscous sum. To measure the fluctuation of the output average,
we replace one draw, calculate its influence on all rows, and add the
squared influences. The count normalization is what makes the resulting
kinetic variance budget decrease with population size.

The convergence proof follows the same stages. Before the thermostat,
the collision speed bound gives strong control of the first force. After
the thermostat, fourth moments handle the unbounded velocities in the
second force. Finally the cap turns velocity transport control into the
fourth-order control needed by the output law. Position noise also deals
with the terminal boundary: a limiting row lands exactly on it with
probability zero. These are estimates for the exact kinetic innovations
conditional on cloning. The randomness of the earlier cloning output
remains a separate part of the complete update.
:::

:::{prf:theorem} Conditional concentration after the coupled second kick
:label: thm-cg-mf-kinetic-variance

Fix the entire post-collision array $(x_i,v_i)_{i=1}^N$ and all parameters
of {prf:ref}`def-cg-mf-parameter-register`. Use count normalization.
Let $\phi(x,v,b)$ satisfy $|\phi|\le M$ and be $L_v$-Lipschitz in
$v$ for each $(x,b)$; no regularity across the terminal boundary is
assumed. Compute the first kick $v_{1i}$ exactly and set

$$
\begin{aligned}
A_4&=8\left[c_h^4\frac1N\sum_i|v_{1i}|^4+q^4d(d+2)\right],\\
A_x&=\sqrt{2/\pi}\,M/s,\\
B_\phi&=aA_x+L_v(1+a^2L_F+2a\nu),\\
C_\phi&=2L_va^2\nu L_\rho,\\
\mathcal B_\phi&=M^2+2B_\phi^2q^2d
                  +8C_\phi^2q^2\sqrt{d(d+2)}\sqrt{A_4}.
\end{aligned}
$$

Then the exact coupled kinetic output satisfies

$$
\operatorname{Var}(L_N^+\phi\mid(x_i,v_i)_i)
\le\mathcal B_\phi/N,
\qquad
\mathbb P\bigl(|L_N^+\phi-\mathbb E[L_N^+\phi\mid(x_i,v_i)_i]|>t
\mid(x_i,v_i)_i\bigr)\le\mathcal B_\phi/(Nt^2).
$$

The budget uses the actual uncapped intermediate velocities. Averaging
over a post-collision law replaces $\sqrt{A_4}$ by
$\sqrt{8[c_h^4V_1^4+q^4d(d+2)]}$, where $V_1$ is the explicit
$p=4$ moment budget of {prf:ref}`lem-cg-mf-moments`. It is independent
of $N$. This theorem controls the kinetic innovations conditional on
cloning; it does not omit the separate variance of the cloning output.
:::

:::{prf:proof}
Integrate only the final position innovation and define
$g(x,u)=\mathbb E_\zeta\phi(x+s\zeta,\psi_V(u),\mathbf1_D(x+s\zeta))$.
The cap is 1-Lipschitz, so $g$ is $L_v$-Lipschitz in $u$.
The total variation distance between Gaussian laws shifted by $r$ is
$2\Phi(r/(2s))-1\le r/(\sqrt{2\pi}s)$; integrating a function bounded
by $M$ multiplies this distance by $2M$. Thus $g$ is $A_x$-Lipschitz
in $x$, including the actual terminal marking.

Conditioned on all OU innovations, the final position draws give
variance at most $M^2/N$. It remains to bound the variance of
$H=N^{-1}\sum_i g(x_{2i},v_{3i})$ under independent OU draws.
Replace only $\xi_j$ by an independent copy $\xi_j'$ and write
$u_i=v_{2i}$, $\Delta u=q(\xi_j'-\xi_j)$ and
$m_1=N^{-1}\sum_i|u_i|$. Only $x_{2j}$ changes directly, by
$a\Delta u$. For $i\ne j$, the viscous force change is at most

$$
\frac\nu N|\Delta u|
\left[1+aL_\rho(|u_j|+|u_i|)\right].
$$

For row $j$ it is at most
$\nu|\Delta u|[1+aL_\rho(m_1+|u_j|)]$.
The conservative force at row $j$ changes by at most
$aL_F|\Delta u|$. Summing the $g$ differences, including the
direct row-$j$ velocity change and both force contributions, gives

$$
|H-H^{(j)}|\le\frac{|\Delta u|}{N}
[B_\phi+C_\phi(m_1+|u_j|)].
$$

The independent-coordinate variance inequality
$\operatorname{Var}H\le\frac12\sum_j\mathbb E|H-H^{(j)}|^2$
is the resampling inequality of
{prf:ref}`lem-chaos-innovation-variance`; its hypotheses hold for
these Gaussian draws. Now
$\mathbb E|\Delta u|^2=2q^2d$ and
$\mathbb E|\Delta u|^4=4q^4d(d+2)$.
Minkowski/Young gives $N^{-1}\sum_j\mathbb E|u_j|^4\le A_4$,
and Jensen gives $\mathbb E m_1^4\le A_4$. Consequently

$$
\frac1N\sum_j\mathbb E[(m_1+|u_j|)^4]\le16A_4,
\qquad
\frac1N\sum_j\mathbb E[|\Delta u|^2(m_1+|u_j|)^2]
\le8q^2\sqrt{d(d+2)}\sqrt{A_4}.
$$

Insert $(B+CY)^2\le2B^2+2C^2Y^2$ in the resampling bound.
Add the final-position conditional variance to obtain $\mathcal B_\phi/N$.
Chebyshev gives the probability bound. Jensen and the explicit fourth
moment budget give the final averaged expression. Dependence created
by the coupled second kick is retained in this calculation.
:::

:::{prf:theorem} Population consistency of the viscous BAOAB stages
:label: thm-cg-mf-kinetic-limit

Use {prf:ref}`def-cg-mf-kinetic-map` with count normalization and the
already specified quadratic reference landscape. Let the
post-collision input arrays have velocities in $\overline B(0,R)$,
empirical laws converging in probability in $W_4$ to deterministic
$\lambda_0$, and a uniform expected position moment of some order $p>4$.
Then their exact finite-$N$ coupled kinetic outputs satisfy

$$
L_N^+\xrightarrow{\mathbb P}
\mathcal K_h^\nu(\lambda_0)\quad\hbox{in }W_4,
$$

where the mark coordinate is given any fixed finite distance scale.
Assume $\partial D$ has Lebesgue measure zero. No restriction that
$\nu$ tend to zero with $N$ is needed for this finite-horizon assertion.
:::

:::{prf:proof}
**First kick and drift.** Couple the input empirical law with
$\lambda_0$. The bounded-velocity estimate of
{prf:ref}`lem-cg-viscous-force-stability` with $p=4$ and the Lipschitz
bound on $F$ show that the law-dependent first B/A map converges in
$W_4$. The moment estimates of {prf:ref}`lem-cg-mf-moments` retain the
uniform $p$th position and velocity moments.

**Thermostat and second drift.** Conditional on this entire intermediate
array, the OU innovations are independent. Apply
{prf:ref}`lem-cg-mf-noise` to $(x_1,v_1)\mapsto
(x_1+a(c_hv_1+q\xi),c_hv_1+q\xi)$. The empirical law at the second
kick converges in $W_4$ to exactly $\lambda_2$.

**Second kick.** Apply the first, unbounded-velocity estimate of
{prf:ref}`lem-cg-viscous-force-stability` to a coupling with $D_4\to0$.
The target fourth velocity moment is finite and fixed. The force
difference tends to zero in $L^2$, and the Lipschitz conservative force
does the same. Thus the joint empirical $(x_2,v_3)$ law converges in
$W_2$; its position marginal still converges in $W_4$. The velocities
at this stage are not declared independent.

**Final noise and cap.** Add an independent Gaussian position innovation
row by row. The bounded-test conditional variance argument still applies
with the entire coupled $(x_2,v_3)$ array as its conditioning variable.
It gives joint weak convergence; the $p$th position bound upgrades the
position component to $W_4$. The 1-Lipschitz cap gives $L^2$ convergence
of coupled capped velocities. Since both capped velocities have norm
at most $V$, $\mathbb E|\Delta v^+|^4\le(2V)^2\mathbb E|\Delta v^+|^2$,
so their fourth-moment transport cost also tends to zero.

**Terminal marks.** The limiting $x^+$ law has density
$(\operatorname{Law}(x_2))*\mathcal N(0,s^2I)$, and therefore assigns
no mass to $\partial D$. Under a position coupling tending to zero,
mark mismatches can occur only near this boundary or when the coupled
position displacement exceeds the chosen tolerance. First send the
displacement error to zero, then shrink the boundary neighborhood.
The bounded mark distance gives convergence of its fourth transport cost.
These component couplings establish the stated marked $W_4$ convergence.
:::

:::{prf:lemma} Local normalization control from the actual population moments
:label: lem-cg-mf-row-local-normalization

Let $\eta$ be a probability law with finite fourth position and velocity
moments, and let $\mu_N$ be an empirical law of $N\ge2$ finite rows.
For a coupling $\pi_N$ of $\mu_N,\eta$, put

$$
\begin{aligned}
D_4&=\left[\int(|x-x'|^2+|v-v'|^2)^2\,d\pi_N\right]^{1/4},\\
M_x&=\left(\int|y|^4\,d\eta\right)^{1/4},\qquad
M_v=\left(\int|w|^4\,d\eta\right)^{1/4},\qquad
m_v=\int|w|\,d\eta,\\
S_\eta&=(1+2M_x^4)^{1/4},\qquad
b_R=\tfrac12\exp[-(R+S_\eta)^2/(2\rho^2)],\\
e_N&=2L_\rho D_4+1/N,\qquad
B_N=(2+4L_\rho M_v)D_4.
\end{aligned}
$$

For every analysis radius $R>0$, the target degree satisfies
$a_\eta(x')\ge b_R>0$ on $|x'|\le R$. No global degree comparison
is assumed. For every $R,H,t>0$, the exact row-normalized force obeys

$$
\begin{aligned}
\pi_N\left(
 \left|\nu\frac{C_{\mu_N}(x,v)}{a_{\mu_N}(x)-1/N}
             -\mathcal V_\eta^{\rm row}(x',v')\right|>t\right)
\le{}&\frac{M_x^4}{R^4}+\frac{M_v^4}{H^4}
       +\frac{4e_N^2}{b_R^2}\\
&+\frac1{t^2}\left[
 \frac{2\nu B_N}{b_R}
 +\frac{2\nu(m_v+H)e_N}{b_R^2}\right]^2.
\end{aligned}
$$

The radii $R,H$ are truncation choices in this probability estimate;
they do not restrict the gas state space or change its kernel.
All constants are the displayed functions of the configured $\nu,\rho$,
the target moments and the actual coupling cost.

*Proof.* Markov's inequality gives
$\eta(|y|>S_\eta)\le M_x^4/(1+2M_x^4)<1/2$.
On $|x'|\le R$ and $|y|\le S_\eta$, the Gaussian kernel is at least
$\exp[-(R+S_\eta)^2/(2\rho^2)]$. Integration over that mass proves
the displayed $b_R$. This lower bound is derived from the law itself.

The Gaussian Lipschitz bound gives, pointwise under the coupling,

$$
|a_{\mu_N}(x)-a_\eta(x')|
\le L_\rho(|x-x'|+W_1(\mu_N^x,\eta^x)).
$$

Consequently
$\|a_{\mu_N}(x)-1/N-a_\eta(x')\|_{L^2(\pi_N)}\le e_N$.
The numerator proof of {prf:ref}`lem-cg-viscous-force-stability`, before
multiplication by $\nu$, gives
$\|C_{\mu_N}(x,v)-C_\eta(x',v')\|_2\le B_N$.
The finite denominator is strictly positive because it is the sum of the
$N-1$ positive nonself kernels divided by $N$.

Exclude $|x'|>R$, $|v'|>H$ and
$|a_{\mu_N}(x)-1/N-a_\eta(x')|>b_R/2$.
Their total coupling mass is at most
$M_x^4/R^4+M_v^4/H^4+4e_N^2/b_R^2$.
On the remaining set both degrees are at least $b_R/2$, the target
degree is at least $b_R$, and $|C_\eta(x',v')|\le m_v+H$.
Subtract the two ratios to obtain

$$
|F_N^{\rm row}-\mathcal V_\eta^{\rm row}|
\le\frac{2\nu}{b_R}|C_{\mu_N}-C_\eta|
  +\frac{2\nu(m_v+H)}{b_R^2}
       |a_{\mu_N}-1/N-a_\eta|.
$$

Minkowski and Chebyshev give the remaining term in the bound.
For fixed $R,H$, that term and the degree-error term tend to zero
when $D_4\to0$, $N\to\infty$. Then $R,H\to\infty$ removes the
two target tails. Thus the force error tends to zero in coupling
probability without a global lower degree. $\square$
:::

:::{prf:theorem} Population consistency of the existing row-normalized kinetic map
:label: thm-cg-mf-row-kinetic-limit

Use the unchanged quadratic reference gas and its existing
$\mathsf n=\mathrm{row}$ configuration bit in
{prf:ref}`def-cg-mf-kinetic-map`. Let its post-collision arrays satisfy
the same inputs as {prf:ref}`thm-cg-mf-kinetic-limit`: velocities in
$\overline B(0,R_c)$ with the actual collision bound
$R_c=(1+2|\alpha_{\rm col}|)V$, empirical laws converging in probability
in $W_4$ to deterministic $\lambda_0$, and a uniform expected position
moment of some order $p>4$. Then the exact finite-$N$ row-normalized
kinetic outputs converge in probability in marked $W_4$ to
$\mathcal K_h^\nu(\lambda_0)$ with row normalization selected.
The configured terminal box has Lebesgue-null boundary.

This statement concerns the existing real-coordinate, exact-Gaussian
mathematical kernel. The native implementation's finite precision and
zero-row-mass underflow mask retain their numerical-error scope. No
global degree assumption, population-dependent force alteration or
new algorithmic cutoff is used. The count-only uncapped B2 moment and
conditional-concentration theorems are not invoked for row normalization.

*Proof.* **Step 1: the bounded-velocity first kick.**
For either the finite nonself row weights or their limiting probability
weights, $|F^{\rm row}(x,v)|\le2\nu R_c$ when all input velocities
have norm at most $R_c$. The limit law $\lambda_0$ has that same
velocity support, since every input empirical law does.
Choose couplings whose fourth cost tends to zero in probability.
The preceding local-normalization lemma gives convergence in coupling
probability of the first force. Its difference has norm at most
$4\nu R_c$, so, for every $t>0$,

$$
\int|\Delta F^{\rm row}|^4d\pi_N
\le t^4+(4\nu R_c)^4\pi_N(|\Delta F^{\rm row}|>t).
$$

First take $N\to\infty$, then $t\downarrow0$. The first-force
fourth transport error vanishes in probability. The configured quadratic
force has $|F(x)-F(x')|\le L_F|x-x'|$. Therefore the first kick
and drift converge in $W_4$ to their row-normalized population maps.
For example their joint fourth transport error is at most

$$
D_4+(1+a)\left[(1+aL_F)D_4
                   +a\|\Delta F^{\rm row}\|_{L^4(\pi_N)}\right],
\qquad a=h/2.
$$

The needed moment bounds follow from the bounded first force, not a
row-normalized population $L^p$ contraction. For
$X_0=(\sup_N\mathbb E[N^{-1}\sum_i|x_i|^p])^{1/p}$ and
$g_{d,p}$ of {prf:ref}`lem-cg-mf-moments`, their explicit budgets are

$$
\begin{aligned}
V_1^{\rm row}&=(1+h\nu)R_c+aL_FX_0+af_0,
&X_1^{\rm row}&=X_0+aV_1^{\rm row},\\
V_2^{\rm row}&=c_hV_1^{\rm row}+qg_{d,p},
&X_2^{\rm row}&=X_1^{\rm row}+aV_2^{\rm row},\\
X_+^{\rm row}&=X_2^{\rm row}+sg_{d,p},
&V_+^{\rm row}&\le V.
\end{aligned}
$$

The actual bounded-donor model supplies
$X_0\le B_D+\sigma_Jg_{d,p}$ and the actual quadratic force has
$f_0=0$. These formulas exhibit every kinetic and landscape dependency;
the selection law supplies its remaining parameter dependencies.

**Step 2: thermostat and second drift.**
The OU innovations are independent conditional on the entire entering
array. The first-stage $W_4$ convergence and the displayed $p>4$
budgets permit {prf:ref}`lem-cg-mf-noise`. The second-stage empirical
law converges in probability in $W_4$ to exactly
$\lambda_2=\operatorname{Law}(x_1+a(c_hv_1+q\xi),c_hv_1+q\xi)$
with the first row-normalized force retained.

**Step 3: second force and terminal cap.**
Apply {prf:ref}`lem-cg-mf-row-local-normalization` with target
$\eta=\lambda_2$. Its fourth moments are finite by Step 2.
The exact denominator $a_{L_N}(x_i)-1/N$ is retained.
For fixed analysis radii the numerator and denominator errors vanish;
letting the radii grow removes the target tails. Thus the coupled B2
force error tends to zero in probability. The input velocities and
quadratic conservative force converge under the same coupling, so
$v_3-v_3'\to0$ in coupling probability. This argument requires no
uniform moment bound on the uncapped row-normalized $v_3$.
The radial cap is continuous, 1-Lipschitz and bounded by $V$, hence

$$
\int|\psi_V(v_3)-\psi_V(v_3')|^4d\pi_N
\le t^4+(2V)^4\pi_N(|v_3-v_3'|>t)\longrightarrow0
$$

after first taking the population limit and then $t\downarrow0$.
Positions at this stage still have their $W_4$ convergence from Step 2.
Consequently the joint empirical law of $(x_2,\psi_V(v_3))$ converges
in probability in $W_4$ and has uniform expected $p$th moments:
the position budget is $X_2^{\rm row}$ and the velocity is capped.

**Step 4: final position noise and marks.**
Final position noise and the velocity-only cap act on different
coordinates, so their observation-law factorization may put the cap
first without changing the configured update. Conditional on the entire
coupled array, apply {prf:ref}`lem-cg-mf-noise` to the fresh position
innovations. Its affine map gives $x^+=x_2+s\zeta$ and the moment
budget $X_+^{\rm row}$, proving joint $W_4$ convergence with the
bounded completed velocities. The limiting position law has a Gaussian
density because the configured $s>0$. Its mass on the actual box boundary
is zero. The boundary-neighborhood coupling argument in
{prf:ref}`thm-cg-mf-kinetic-limit` therefore makes terminal mark
mismatches vanish. This proves the marked fourth transport convergence
for the exact row-normalized kinetic stages. $\square$
:::

(sec-cg-mf-full-update)=
## 4. Cloning, finite horizons, and recorded geometry

:::{div} feynman-prose
We can now join the kinetic calculation to the collision law already
proved for the canonical gas. That law remembers the random component
that changes a tagged walker's velocity; it does not replace collision
partners by independent output walkers. Revival puts every entering row
at an eligible donor position, with the configured jitter, so the
post-collision moment estimates remain available even when retained dead
positions were far away.

There is also a finite-population event to account for: the whole swarm
can die. The limiting law's positive alive fraction makes that event
vanish over any fixed number of updates, as the proof shows. This allows
a harmless observation convention after extinction without introducing
a restart. It also relates the original path law to conditioning once
on survival through that horizon. Finally, an empirical population law
and a recorded graph answer different questions. The completed-state
chaos consequence uses exchangeability. Tagged histories and graph
descriptors involving more and more rows still need the estimates for
those particular joint observations.
:::

:::{prf:corollary} Fixed-horizon population limit of the recorded viscous gas
:label: cor-cg-mf-full-update

Consider the quadratic reference recorded gas of
{prf:ref}`def-variant-recorded-color-geometry` on the canonical bounded
terminal domain, with either of its existing Gaussian force normalizations
fixed throughout the recursion. Keep its configured reward and donor rule fixed. Assume
the measurement-consistency hypotheses of
{prf:ref}`thm-mean-field-one-step-consistency` at the initial law, positive
alive mass, and the initial convergence and uniform integrability needed
there. Let $\tau_\dagger$ be the first completed update with no alive
slot, including update zero if the initial swarm is extinct. Use the
actual empirical marked law on $\{\tau_\dagger>n\}$ and the fixed
probability $\delta_{(0,0,0)}$ at and after the cemetery state. This is
an observation convention for the absorbing extension; it supplies no
restart donor. Then at every fixed finite number of updates this
extended empirical marked law converges in probability to the uniquely
specified recursion

$$
\mu_{n+1}=\mathcal F_h^\nu(\mu_n).
$$

For each fixed horizon $T$,
$\Pr(\tau_\dagger\le T)\to0$. The original complete path law
conditioned once on survival through that entire horizon has the same
limit. This statement does not condition and renormalize each
intermediate transition separately.

For exchangeable input laws the fixed-row marginals converge to
$\mu_n^{\otimes l}$ for every fixed $l$. The physical-time convention is
$\tau_n=nh_{\rm phys}=nt_*h$. Existence and uniqueness here concern this initial-value
recursion, not a unique stationary phase.

The fixed-row assertion concerns completed states only. Empirical
convergence alone does not control a chosen deterministic row or its
intermediate-stage history: an exceptional row can have vanishing
empirical mass. An ordered-star implementation must retain its priority
state and cannot invoke exchangeability of the Haar component kernel.
Descriptor transfer applies to precisely the joint variables whose
convergence has been established, by continuous mapping (or at a
discontinuity set of limiting probability zero). Multi-stage tagged
histories, increasing-row graph descriptors and continuum curvature
remain separate estimates; recording them does not prove their limits.
:::

:::{prf:proof}
The selection/collision mechanism is unchanged. Apply its one-step
consistency theorem to obtain weak post-collision empirical convergence.
Bounded donor positions, Gaussian jitter and the deterministic collision
velocity bound give uniform moments of all orders after collision.
Truncation upgrades this to $W_4$ convergence. Apply
{prf:ref}`thm-cg-mf-kinetic-limit` for count normalization or
{prf:ref}`thm-cg-mf-row-kinetic-limit` for row normalization to complete
the update. The latter uses the bounded first-kick force and the capped
completed velocity; no uncapped B2 row-moment estimate is imported from
the count-normalized theorem.

For the canonical terminal domain with nonempty interior, final position noise gives
$\mathbb P(x^+\in D)>0$. Its fourth moment is finite and its boundary
mass is zero. These are the alive-mass and moment inputs to the next
selection step.

Initially, convergence of the empirical alive fraction to $m(\mu_0)>0$
implies $\Pr(\tau_\dagger=0)\to0$. Inductively condition on survival
through update $n$, whose probability tends to one. The empirical input
still converges to $\mu_n$ under this conditioning. The one-step theorem
for random inputs and the kinetic limit give convergence of the proposed
output alive fraction to $m(\mu_{n+1})>0$. Hence

$$
\Pr(\tau_\dagger=n+1\mid\tau_\dagger>n)
\le\Pr\left(\left|m(L_N^{n+1,\rm raw})-m(\mu_{n+1})\right|
       \ge\tfrac12m(\mu_{n+1})\ \middle|\ \tau_\dagger>n\right)
\longrightarrow0.
$$

Here $L_N^{n+1,\rm raw}$ is the actual marked output before replacing
an extinct output by the cemetery observation. This replacement occurs
with probability tending to zero, so it does not change convergence in
probability. Summing over the finitely many updates proves the asserted
extinction limit and completes the induction.

For any bounded complete-path test $A$, conditioning on
$\{\tau_\dagger>T\}$ changes its expectation by at most
$2\|A\|_\infty\Pr(\tau_\dagger\le T)$, by splitting the expectation
over survival and extinction. The exact total-variation identity of
{prf:ref}`thm-chaos-conditioned-propagation` is the same conditioning
calculation; its canonical zero-viscosity extinction rate is not used
here. Thus whole-horizon survivor conditioning has the same finite-horizon
limit without changing this viscous transition.

For bounded tests, expansion of sampling with and without replacement shows that
exchangeability and convergence of the empirical law to a deterministic
law imply convergence of any fixed-row product test to its product
expectation; the without-replacement error is $O(l^2/N)$. Product tests
determine the marginal law. Continuous mapping proves only descriptors
of these convergent completed-state marginals, with the stated
null-discontinuity restriction. No intermediate tagged-path assertion
is used in the proof.
:::

:::{prf:proposition} Particle-number-independent conditional force sampling error
:label: prop-cg-mf-force-sampling

For iid samples $(X_i,V_i)$ from $\lambda$ with $|V_i|\le R$, the
count-normalized force at a tagged row satisfies

$$
\mathbb E\left[
 |F_i^{\mathrm{visc},N}-\mathcal V_\lambda(X_i,V_i)|^2
 \ \middle|\ X_i,V_i\right]
\le\frac{4\nu^2R^2}{N}.
$$

For arbitrary dependent rows, a bounded empirical test at the fresh OU
stage has the conditional $N^{-1}$ variance bound of
{prf:ref}`lem-cg-mf-noise`. This proposition supplies the force-sampling
term for an independent comparison population; coupling error and
cloning bias must still be added for the actual interacting population.
:::

:::{prf:proof}
Conditional on the tagged row, the $N-1$ other vectors
$Y_j=K_\rho(X_i,X_j)(V_j-V_i)$ are iid, with $|Y_j|\le2R$ and mean
$m$. The force estimator is $\nu N^{-1}\sum_{j\ne i}Y_j$.
Its bias is $-\nu m/N$. Its conditional mean squared error is

$$
\frac{\nu^2}{N^2}
 \bigl[(N-1)(\mathbb E|Y_j|^2-|m|^2)+|m|^2\bigr]
\le\frac{\nu^2}{N}\mathbb E|Y_j|^2
\le\frac{4\nu^2R^2}{N}.
$$

The noise-stage statement conditions on the full array and follows
from the independent fresh innovations, irrespective of dependence
in that array.
:::

(sec-cg-mf-open)=
## 5. Stationary and geometric obligations retained downstream

:::{div} feynman-prose
The finite-horizon result tells us how a specified initial population
evolves for a fixed number of updates. Following that population for
arbitrarily long times requires another calculation. Selection can
respond to the law it helps create, and that feedback can support
different stationary phases. The viscous terms and the remaining
within-cell contributions must therefore stay inside any proposed
entropy or attraction estimate.

Geometry poses a related question. Recording neighbors and metric
payloads gives exact observations of a finite run. To claim a continuum
metric or curvature action, we must control how those observations
change as the population and graph scales change together. A theorem
about bounded tests at a fixed scale does not supply the errors for
shrinking neighborhoods or changing tessellations. The statements below
identify these two remaining tasks within the same configured model.
The proved finite-population and population-limit results remain usable
inputs, but they do not determine stationary selection, a uniform
full-gradient inequality, or geometric first variations by themselves.
:::

:::{prf:conjecture} Population-uniform stationary control of the recorded viscous gas
:label: conj-cg-mf-stationary-control

For a declared active-cloning parameter regime, extend the signed
population entropy calculation of
{prf:ref}`rem-slc-completion` to the exact map $\mathcal F_h^\nu$.
The desired conclusion is phase-specific attraction, stationary chaos
and an adequate full-gradient inequality with constants independent of
$N$. Neither the fixed-horizon theorem above nor fixed-$N$ QSD
uniqueness establishes these statements. The missing estimate is
coercivity of the signed population feedback, with the viscous and
within-cell residual terms retained.
:::

:::{prf:conjecture} Same-record geometry consistency for the physical variant
:label: conj-cg-mf-geometry-consistency

For the chosen recorded spatial graph, prove the conditional sampling,
correlation, metric error and same-sample action estimates required by
{prf:ref}`cor-continuum-consistency-conditional` on one declared schedule.
An actual full-law LSI may supply covariance estimates after compatible
dynamics and sampling are specified. The finite-horizon particle limit
does not by itself control shrinking-bandwidth tests, Delaunay changes,
curvature, or first variations of the geometry action.
:::
