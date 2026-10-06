# Weighted population feedback for the active count-viscous law

This note treats the deterministic population providers of the conservative
mean-field gas. The completed estimates retain the joint OU law of landing
position and velocity. Their combination proves convergence of the active
conservative population law for an explicit positive viscosity and positive
fitness-power interval at the original harmonic step size. The finite-swarm
empirical providers are random and correlated with the observed row; the
population formulas here do not replace their joint-array analysis.

(sec-pvb-regime)=
## 1. A nonresonant conservative population regime

:::{div} feynman-prose
At a small time step a harmonic force leaves a small fraction of the entering
position in the next position. This gives a drift estimate, rather than the
exact reset used in the earlier smoothing note. We keep that fraction and use
the drift to find a region where Gaussian noise gives a common probability
for every frozen population provider.

The two viscous providers have different roles. The first is the prepared
phase law. The second is its joint noisy landing-position/OU-velocity law.
In particular the second provider cannot be assembled from two independent
marginals.
:::

:::{prf:definition} Nonresonant population record
:label: def-pvb-record

Use the canonical all-alive conservative population preparation of
{prf:ref}`def-kv-smoothing-regime`, with $F(x)=-x$, and a bounded $C^1$
reward of oscillation $R_{\rm osc}$ and spatial derivative bound $L_R$.
Retain both positive fitness powers, the actual population reward/diversity
normalizers, Gaussian recipient jitters, and the full rooted component
collision law. Set
$$
t=h/2,\quad c=e^{-\gamma h},\quad b=t(1+c),\quad
m=1-t^2>0,\quad a_x=1-tb\in(0,1),\quad
\tau^2=t^2q^2+s^2,
$$
with $q,s,V,\rho,\sigma_J>0$. Write $V_c=(1+2|\alpha_{\rm col}|)V$.
For a prepared population $\lambda$ define the deterministic fields
$$
a_0(x)=\int K_\rho(x,x')\,\lambda(dx',dv'),\quad
m_0(x)=\int K_\rho(x,x')v'\,\lambda(dx',dv'),\quad
C_0(x,v)=m_0(x)-a_0(x)v.
$$
The first stages are
$$
U=v+t\nu C_0(x,v),\quad u=U-tx,\quad
x_1=mx+tU,\quad w=cu+q\xi,\quad y=x_1+tw.
$$
Let $\Lambda_2$ be this actual joint $(y,w)$ law. Its fields are
$a_2(y)=\int K_\rho(y,y')\Lambda_2(dy',dw')$ and
$m_2(y)=\int K_\rho(y,y')w'\Lambda_2(dy',dw')$.
The actual second kick and final operations are
$$
z=(1-t\nu a_2(y))w-ty+t\nu m_2(y),\quad
x^+=y+s\zeta,\quad v^+=C_V(z).
$$
The Gaussian innovations are independent within each own law. The second
fields are deterministic population fields of the joint law $\Lambda_2$.

Put $W_p(x)=1+|x|^p$ and
$\|\xi\|_{W_p}=\int W_p|\xi|$ for signed measures. The full variation
convention is used throughout this note.
:::

:::{prf:lemma} Invariant eighth-moment class for active count viscosity
:label: lem-pvb-eighth-moment

Set
$$
r_8=(1+a_x^8)/2,\quad
e_8=(r_8/a_x^8)^{1/7}-1,\quad
D_8=bV_c+\sqrt{a_x^2\sigma_J^2+\tau^2}\,G_8^{1/8},
$$
$$
B_8=(1+e_8^{-1})^7D_8^8,\qquad
H_8=2B_8/(1-r_8).
$$
If $0\le t\nu\le1$ and
$c_*=a_*/\kappa_C\le(1-r_8)/(2r_8)$, then the actual population map
preserves $\mathfrak C_8=\{\mu: \mu|x|^8\le H_8,\ |v|\le V\}$.
Every output also has the fresh position-Gaussian representation used in
{prf:ref}`thm-kv-active-joint-bv`.
The same bound holds for the averaged eighth moment of every actual finite
swarm; no population limit is needed for that assertion.
:::

:::{prf:proof}
The simultaneous source law retains each root position or copies a donor.
Its accepted donor density is at most $c_*$ times the base population law;
in a finite swarm its incoming column sum is at most $c_*$. Thus the
source eighth moment, averaged over roots, is at most
$(1+c_*)\mu|x|^8$. Component collision velocities obey $V_c$ and the
first count velocity average is convex. Conditional on the actual source,
the landing position is $a_x x_{\rm source}+bU+a_xI J+tq\xi+s\zeta$.
The Gaussian covariance in this expression is at most
$(a_x^2\sigma_J^2+\tau^2)I_d$; the bounded part has norm at most $bV_c$.
The scalar Young inequality
$(A+B)^8\le(1+e)^7A^8+(1+e^{-1})^7B^8$
therefore gives
$$
M_8^+\le r_8(1+c_*)M_8+B_8.
$$
The gate condition bounds its coefficient by $(1+r_8)/2$ and closes
the sublevel $H_8$. The final Gaussian is drawn after storing the capped
velocity, giving the stated representation.
:::

(sec-pvb-weighted-score)=
## 2. Weighted BV supplied by the actual preparation

:::{div} feynman-prose
The weight follows large positions through a comparison. We compute it
inside the Gaussian change of variables. An unweighted density derivative
and a separate moment bound would not justify the weighted derivative used
later.
:::

:::{prf:lemma} Explicit weighted spatial variation of preparation
:label: lem-pvb-weighted-preparation-bv

Let an entering law have fresh Gaussian positions of variance $s^2I_d$,
stored speed at most $V$, and moment $M_p=\mathbb E|x|^p<\infty$,
$p\ge1$. Use bounded reward in {prf:ref}`def-pvb-record` and define
$$
K_r=2/\sigma_r+2R_{\rm osc}/(3\sqrt3\,\sigma_r^2),\qquad
K_s=2/\sigma_s+2S_b/(3\sqrt3\,\sigma_s^2),
$$
$$
B_{\rm pat}^{\rm bd}=B_D+2a_*B_C+
2(L_{\rm rec}+\kappa_C^{-1}L_{\rm don})
       [H_rK_rL_R+H_sK_s(1+\kappa_D^{-1})],
$$
where $B_D,B_C$ are (KVS.2). Put
$$
\begin{aligned}
L_{G,p}&=\frac{g_1}{s}(1+2^{p-1}M_p)
                    +2^{p-1}s^{p-1}G_{p+1},\\
L_{J,p}&=\frac{g_1}{\sigma_J}(a_*+2^{p-1}c_*M_p)
                    +2^{p-1}a_*\sigma_J^{p-1}G_{p+1},\\
B_p^{\rm src}&=d\left[L_{G,p}+L_{J,p}
       +\frac{2c_*g_1}{\sigma_J}(1+M_p)
       +(1+M_p)B_{\rm pat}^{\rm bd}\right].
\end{aligned}
\tag{PVB.1}
$$
The actual prepared sampled-row law $\lambda$ has finite spatial derivative
measures with
$$
\sum_a\int (1+|X|^p)|D_{X_a}\lambda|\le B_p^{\rm src}.
\tag{PVB.2}
$$
The finite-swarm version bounds the average over rows of their respective
weighted joint-array derivatives by the same constant. Hence (PVB.2) also
holds for population limits of these actual preparations with the stated
uniform moments.
:::

:::{prf:proof}
Repeat the branchwise changes of variables in
{prf:ref}`thm-kv-active-joint-bv`. On a persistent row, its prepared
position is $X_i=x_i=Y_i+sZ_i$. Jensen conditional on $Y_i$ gives
$\mathbb E|Y_i|^p\le\mathbb E|x_i|^p$. The derivative of the entering
Gaussian, with its actual positional weight, is bounded by
$$
\mathbb E(1+|Y_i+sZ_i|^p)|Z_{i,a}|/s
\le\frac{g_1}{s}(1+2^{p-1}\mathbb E|Y_i|^p)
                   +2^{p-1}s^{p-1}G_{p+1}.
$$
Its average is $L_{G,p}$. Each compensation jitter has derivative
integral $g_1/\sigma_J$. The conditional accepted incoming count is at
most $c_*$; retaining the larger $2c_*$ gives the third term of (PVB.1).

On a copied row, $X_i=x_j+\sigma_JZ_i$. Its own jitter derivative obeys
$$
\mathbb E(1+|x_j+\sigma_JZ_i|^p)|Z_{i,a}|/\sigma_J
\le\frac{g_1}{\sigma_J}(1+2^{p-1}|x_j|^p)
                   +2^{p-1}\sigma_J^{p-1}G_{p+1}.
$$
The total accepted mass is at most $a_*$ and its donor column density
at most $c_*$; averaging gives $L_{J,p}$.

The pattern-weight derivative is present only on the persistent branch.
There its output weight is $1+|x_i|^p$. The bounded-reward proof in
{prf:ref}`thm-kv-active-joint-bv` bounds the conditional derivative by
the displayed bounded-reward coefficient. The only random count in the
diversity derivative is its incoming measurement count; conditional on the
entering positions its expectation is at most $\kappa_D^{-1}$.
Multiplying by $1+|x_i|^p$, then averaging, gives
$(1+M_p)B_{\rm pat}^{\rm bd}$. All Gaussian innovations used in the
compensations are independent of the discrete pattern.

These calculations bound the weighted distributional derivative measures
on the full array. Pushforward to the selected row gives (PVB.2).
Weak convergence, tested first against compactly supported derivative
tests and then increasing truncations of the nonnegative weight, preserves
the bound in a population limit. No factorization of its prepared law is
introduced.
:::

:::{prf:lemma} Complete weighted joint OU scores
:label: lem-pvb-joint-scores

Suppose $0\le\nu\le\bar\nu$, where
$$
L_0=4dV_c\ell_\rho,\quad L_2=8dV_c/\rho^2,\qquad
\bar\nu=\min\{1,m/(2t),m/(2t^2L_0)\}>0.
$$
For a prepared input with moment $M_{\rm p,5}$ and weighted spatial
variation $B_5^{\rm src}$, set $L_I=(m-t^2\nu L_0)^{-1}$,
$D_1=t+t\nu L_0$, and
$$
H=1+a_x+bV_c+tq+ct+cV_c+q,\qquad C_*=7\cdot3^4H^5,
$$
$$
\begin{aligned}
M_*&=C_*(1+G_5)(1+M_{\rm p,5}),\\
S_x&=C_*\left[L_I(1+G_5)B_5^{\rm src}
 +dL_I^2t^2\nu L_2(1+G_5)(1+M_{\rm p,5})
 +\frac{dcD_1L_I}{q}(G_1+G_6)(1+M_{\rm p,5})\right],\\
S_w&=\frac{dC_*}{q}(G_1+G_6)(1+M_{\rm p,5}).
\end{aligned}
\tag{PVB.3}
$$
The actual joint density $f(x_1,w)$ has
$$
\int \mathcal W_*(y,w)f\le M_*,\quad
\int \mathcal W_*(y,w)|\nabla_{x_1}f|_1\le S_x,\quad
\int \mathcal W_*(y,w)|\nabla_w f|_1\le S_w,
\tag{PVB.4}
$$
where $y=x_1+tw$ and
$\mathcal W_*=1+|y|^4+|w|^4+(1+|y|^4)(1+|w|)$.
For $\rho(y,w)=f(y-tw,w)$,
$$
\int(1+|y|^4)(1+|w|)|\nabla_w\rho|_1\le S_w+tS_x.
\tag{PVB.5}
$$
The same constants apply to interpolation between any two first providers
whose velocities are supported in $\overline B_{V_c}$. The derivative
bounds also hold with absolute values taken in each prepared-velocity
fibre before integrating its fixed mixing measure. Thus no cancellation
between different velocity fibres is needed.
:::

:::{prf:proof}
The first position map at fixed $v$ is
$x\mapsto mx+tv+t^2\nu C_0(x,v)$. Its global inverse and coordinate
Jacobian estimates are exactly the proof of
{prf:ref}`thm-kv-count-joint-density`, with $A$ replaced by $m$ and
$\lambda$ by one. The same overestimates $L_0,L_2$ remain valid for
the deterministic provider integral.

Write $w=cU-ctX+q\xi$ and $y=a_xX+bU+tq\xi$.
Since $|U|\le V_c$, both $|y|$ and $|w|$ are at most
$H(1+|X|+|\xi|)$. Every term of $\mathcal W_*$ is bounded by
$7H^5(1+|X|+|\xi|)^5$, and
$$
(1+|X|+|\xi|)^5\le3^4(1+|X|^5+|\xi|^5)
\le3^4(1+|X|^5)(1+|\xi|^5).
$$
This proves the $M_*$ bound. The pushed source derivative is bounded by
$L_I$ times its weighted source variation, integrated against
$C_*(1+|\xi|^5)$. It gives the first term of $S_x$.
The logarithmic determinant derivative has coordinate-sum bound
$dL_I^2t^2\nu L_2$ and gives its second term. The derivative of the
Gaussian mean has coordinate matrix bounds $cD_1L_I$; multiplying its
Gaussian score by $\mathcal W_*$ uses
$\mathbb E[(1+|\xi|^5)|\xi|]=G_1+G_6$.
The larger coordinate-sum factor $d$ gives the last term of $S_x$ and
the bound $S_w$. The prepared velocity mixing measure can remain singular.
Spatial mollification and weighted truncation justify these calculations
for BV inputs, followed by lower semicontinuity as in that theorem.
Finally $\nabla_w\rho=(\nabla_wf-t\nabla_{x_1}f)(y-tw,w)$.
The weight in (PVB.5) is one of the nonnegative terms of $\mathcal W_*$,
which proves the shear estimate. Convex interpolation of providers
preserves all field bounds used here.
:::

(sec-pvb-feedback)=
## 3. Absolute viscous provider feedback

:::{div} feynman-prose
We can now charge the change of viscous fields while holding one compared
prepared law fixed. The first change moves its drift coordinates. The second
change rescales and translates velocity at each fixed landing position.
The calculation uses the correlated density of that fixed law throughout.
:::

:::{prf:theorem} Quantitative feedback of both deterministic count providers
:label: thm-pvb-viscous-feedback

Let $\lambda,\widetilde\lambda$ be prepared laws with speed bound $V_c$,
and let $\widetilde\lambda$ satisfy (PVB.2)--(PVB.4), uniformly under
first-provider interpolation. Let $\Lambda_2,\widetilde\Lambda_2$ be
their actual respective joint stage laws. Put
$D=\|\lambda-\widetilde\lambda\|_{W_4}$ and
$$
\begin{aligned}
C_T&=2V_c[t^2S_x+ctS_w+dt^2\ell_\rho L_IM_*],\\
C_{
\rm stage}&=1+27[a_x^4+(ct)^4+(bV_c)^4+(cV_c)^4
                         +((tq)^4+q^4)G_4],\\
C_{
\rm out}&=1+8(1+s^4G_4),\\
C_{B2}&=\frac{C_{\rm out}}{1-t\nu}
                    [dM_*+2(S_w+tS_x)].
\end{aligned}
\tag{PVB.6}
$$
Then
$$
\|\Lambda_2-\widetilde\Lambda_2\|_{1+|y|^4+|w|^4}
\le(C_{\rm stage}+\nu C_T)D.
\tag{PVB.7}
$$
Let $K[\lambda,\Lambda_2]$ denote the kinetic kernel with these frozen
providers, including its actual smooth cap and independent final position
Gaussian. Holding the entering prepared law $\widetilde\lambda$ fixed,
$$
\left\|\widetilde\lambda K[\lambda,\Lambda_2]
 -\widetilde\lambda K[\widetilde\lambda,\widetilde\Lambda_2]\right\|_{W_4}
\le\nu C_{\rm fb}D,
\quad
C_{\rm fb}=C_{\rm out}C_T+
                  C_{B2}(C_{\rm stage}+\nu C_T).
\tag{PVB.8}
$$
No conditional finite-swarm empirical provider appears in this theorem.
:::

:::{prf:proof}
For a first-provider difference, the velocity-supported weighted norm gives
$\|\Delta C_0\|_\infty\le2V_cD$ and
$\|D_x\Delta C_0\|\le2V_c\ell_\rho D$.
Interpolate the first provider and hold $\widetilde\lambda$ fixed. In
the $(x_1,w)$ coordinates its Eulerian displacement has components
$$
A_\theta=t^2\nu\Delta C_0(X_\theta,v),\qquad
B_\theta=ct\nu\Delta C_0(X_\theta,v),
$$
where $X_\theta$ is the inverse first drift at fixed $x_1,v$.
For each prepared velocity fibre, the derivative of its joint density is
$-\operatorname{div}_{x_1}(A_\theta f_\theta)
-\operatorname{div}_w(B_\theta f_\theta)$.
The second displacement is independent of $w$ on that fibre, so its
$w$-divergence is zero. The first divergence has absolute bound
$2dV_ct^2\nu\ell_\rho L_ID$. Apply (PVB.4), integrate the velocity
fibres, then the interpolation parameter. The weighted stage change is
at most $\nu C_TD$. Weighted truncations justify the divergence formula
for BV input, with its explicitly integrable scores as dominators.

At a fixed first provider, the stage Markov map satisfies
$$
\mathbb E[1+|y|^4+|w|^4\mid X,v]\le C_{\rm stage}(1+|X|^4)
$$
by $|U|\le V_c$, the two displayed affine identities, and
$(A+B+C)^4\le27(A^4+B^4+C^4)$. Pushforward of a signed input measure
therefore has weighted norm at most $C_{\rm stage}D$. Combining the
fixed-provider input difference with its field difference gives (PVB.7).

Now hold the actual correlated stage density $\rho(y,w)$ of
$\widetilde\lambda$ fixed, and interpolate the second provider.
Write $l_\theta(y)=1-t\nu a_{2,\theta}(y)\ge1-t\nu$.
At each fixed $y$, the affine velocity map has Eulerian pullback
displacement
$$
\frac{t\nu}{l_\theta(y)}[\Delta m_2(y)-w\Delta a_2(y)].
$$
Its divergence in $w$ contributes at most
$dt\nu|\Delta a_2(y)|/l_\theta(y)$ times the density. Its derivative
term contributes at most
$t\nu[|\Delta m_2(y)|+|w||\Delta a_2(y)|]
|\nabla_w\rho|_1/l_\theta(y)$.
The stage norm in (PVB.7) bounds both $\|\Delta a_2\|_\infty$ and
$\|\Delta m_2\|_\infty$: $K_\rho\le1$ and
$|w|\le1+|w|^4$.

The final cap is a common Markov pushforward. The final Gaussian gives
$\mathbb E[W_4(y+s\zeta)]\le C_{\rm out}(1+|y|^4)$, independently
of its pre-cap velocity. Thus (PVB.4)--(PVB.5) bound this second-provider
change by $\nu C_{B2}\|\Lambda_2-\widetilde\Lambda_2\|$
in the stage weighted norm. In particular $\rho$ is never replaced by
the product of its marginals.

Change first providers while holding the second fixed, then change the
second provider while holding the own first stage fixed. The first change
has postprocessed norm at most $\nu C_{\rm out}C_TD$. The second has
the bound just proved. Substitute (PVB.7) to obtain (PVB.8).
:::

(sec-pvb-frozen-harris)=
## 4. Uniform weighted contraction for frozen whole-update providers

:::{div} feynman-prose
A velocity map can fold and still give a lower density bound. We need every
target velocity in a small ball to have a preimage with controlled Gaussian
probability; we do not need that preimage to be unique. The positive linear
part makes the map proper and gives degree one. Counting its regular
preimages in the change-of-variables formula then gives the common density.
:::

:::{prf:lemma} Local phase minorization for the frozen full active root kernel
:label: lem-pvb-frozen-minorization

Freeze an entering population $\mu\in\mathfrak C_8$, its actual
preparation-provider kernel $J_\mu$, and its two actual viscous providers.
Let $P_\mu$ be this frozen full root kernel. Require $2c_*<1$ and
$0\le t\nu<m$. Suppose the joint second provider has
$\Lambda_2|w|\le M_2$. On $|x|\le R_x$, define
$$
R_1=mR_x+tV_c,\quad M_v=cV_c+ctR_x,\quad
Q=\frac{u+tR_1+t\nu M_2}{m-t\nu},\quad
L_z=m+t\nu+t^2\nu\ell_\rho(Q+M_2),
$$
for declared $u,r>0$, and
$$
k_v=(2\pi q^2)^{-d/2}e^{-(Q+M_v)^2/(2q^2)}L_z^{-d},\quad
k_x=(2\pi s^2)^{-d/2}e^{-(r+R_1+tQ)^2/(2s^2)},
$$
$$
\epsilon=(1-a_*)e^{-c_*}v_d(u)v_d(r)k_vk_x>0.
\tag{PVB.9}
$$
Then $P_\mu(z,\cdot)\ge\epsilon\vartheta$ on that input set,
where $\vartheta$ is uniform position on $B_r$ times the cap-pushforward
of uniform pre-cap velocity on $B_u$. The common probability and floor
are uniform over all providers with the stated bounds.
:::

:::{prf:proof}
In the actual frozen rooted component law, conditional on its root type,
the outgoing acceptance probability is at most $a_*$ and its incoming
Poisson intensity at most $c_*$. The outgoing token and incoming process
are independent conditional on that type. Therefore a persistent isolated
root has conditional probability at least $(1-a_*)e^{-c_*}$. On this
event the root preparation preserves $(x,v)$ exactly.

For this input, $|U|\le V_c$, $|x_1|\le R_1$, and the Gaussian mean of
$w$ has norm at most $M_v$. At fixed $x_1$, its second-kick map is
$$
Z(w)=[m-t\nu a_2(x_1+tw)]w-tx_1+t\nu m_2(x_1+tw).
$$
Since $0\le a_2\le1$ and $|m_2|\le M_2$,
$|Z(w)|\ge(m-t\nu)|w|-tR_1-t\nu M_2$.
The map is proper. The homotopy replacing $t\nu$ by $\theta t\nu$
has the same coercivity bound for $0\le\theta\le1$. On a sufficiently
large sphere it has no preimage of a fixed target in $B_u$; its degree
equals that of $mw-tx_1$, namely one. It is therefore onto, and all
preimages of targets in $B_u$ lie in $B_Q$.

The Gaussian convolution fields are smooth, with
$|Da_2|\le\ell_\rho$ and $|Dm_2|\le\ell_\rho M_2$.
On $B_Q$, differentiation gives $\|DZ\|\le L_z$.
The map is $C^\infty$ between two $d$-dimensional Euclidean spaces, so
almost every target is a regular value by
[Sard's theorem](https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-48/issue-12/The-measure-of-the-critical-values-of-differentiable-maps/bams/1183504867.pdf).
Its nonempty
preimage set is a finite set in $B_Q$: regular preimages are isolated and
the set is compact. The area formula expresses the absolutely continuous
part of its Gaussian pushforward as a sum over those preimages, each with
denominator $|\det DZ|$. A possible singular part is nonnegative and
preserves the ensuing lower measure bound.
At least one summand is at least $k_v$. Conditional on a preimage,
the final position Gaussian has center $x_1+tw$ of norm at most
$R_1+tQ$, so its density on $B_r$ is at least $k_x$.
This gives the joint density floor before capping. The common cap
pushforward preserves the measure inequality. Multiply by the isolated
root probability to obtain (PVB.9).
:::

:::{prf:theorem} Explicit contraction for frozen population-provider kernels
:label: thm-pvb-frozen-weighted-contraction

For every frozen $P_\mu$ above, suppose its entering donor provider $\mu$
has fourth moment at most $H_4$ and $0\le t\nu\le1$.
Set $r_4=(1+a_x^4)/2$, $e_4=(r_4/a_x^4)^{1/3}-1$,
$$
B_4=(1+e_4^{-1})^3
       [bV_c+\sqrt{a_x^2\sigma_J^2+\tau^2}\,G_4^{1/4}]^4,
\quad B=1-r_4+r_4c_*H_4+B_4.
$$
Choose $R>\max\{2,2B/(1-r_4)\}$ and use (PVB.9) with
$R_x=(R-1)^{1/4}$. Choose
$0<\beta<2\epsilon/(r_4R+2B)$ and put
$$
w_\beta=1+\beta W_4,\quad
q_H=\max\left\{
\frac{2+\beta(r_4R+2B)}{2+\beta R},\quad
1-\epsilon+\frac\beta2(r_4R+2B)\right\}<1.
\tag{PVB.10}
$$
Then, for every zero-mass signed measure $\xi$,
$$
\|\xi P_\mu\|_{w_\beta}\le q_H\|\xi\|_{w_\beta}.
\tag{PVB.11}
$$
All constants can be chosen uniformly over the declared provider moment
class and a fixed positive viscosity interval satisfying the preceding
inequalities.
:::

:::{prf:proof}
The conditional source fourth moment at a root $z$ is at most
$|x|^4+c_*H_4$. Repeat the Gaussian/Young calculation of
{prf:ref}`lem-pvb-eighth-moment` at exponent four to obtain
$P_\mu W_4(z)\le r_4W_4(z)+B$.

Use the ground cost
$d_\beta(z,z')=0$ for equality and
$d_\beta(z,z')=2+\beta[W_4(z)+W_4(z')]$ otherwise.
For distinct inputs with $W_4(z)+W_4(z')\ge R$, any coupling of their
outputs has expected cost at most
$2+\beta[r_4(W_4(z)+W_4(z'))+2B]$.
Its ratio to the input cost is at most the first term of $q_H$;
this follows by differentiating the ratio in the sum of the input weights.
If their weight sum is below $R$, both are in the minorization set.
Couple its common probability $\epsilon\vartheta$ identically and couple
the residual probabilities in any way. The expected cost is at most
$2(1-\epsilon)+\beta(r_4R+2B)$, whose ratio to the input cost is at
most the second term of $q_H$. Identical inputs use the identical output
coupling. The two strict inequalities in (PVB.10) follow from the choices
of $R,\beta$.

The optimal transport cost for $d_\beta$ equals
$\int w_\beta|\mu-\mu'|$: match their common part identically, and
the remaining disjoint measures pay their two weights. For the lower
bound, every unmatched mass must pay those weights. Integrating the
constructed kernel coupling against this input coupling proves the
weighted variation contraction for probability differences. Scaling the
positive and negative parts proves (PVB.11) for every zero-mass signed
measure.
:::

(sec-pvb-preparation-closure)=
## 5. Completed active population convergence

:::{div} feynman-prose
The missing preparation estimate is now supplied by exposing the tagged
root's source before exploring its remaining component. This keeps a large
donor weight attached to that source while the rest of the component is
compared. Its coefficient is proportional to the positive fitness powers.

The two small parameters have separate jobs. Small powers control
preparation feedback; small viscosity controls the change of its two
population fields. Both fit inside the strictly positive mixing margin of
the frozen kernel. Neither estimate asks the realized fitness variance to
exceed a fixed threshold.
:::

:::{prf:definition} Explicit uniform envelopes for the positive parameter interval
:label: def-pvb-positive-envelopes

Use the primitive linear envelopes of
{prf:ref}`lem-kuhw-positive-preparation-register`:
$a_*\le\theta a_0$, $c_*\le\theta c_0$,
$H_b\le\theta J_b$, and $L_{\rm rec},L_{\rm don}\le G_0$ for
$0<\theta\le1$, with $c_0=a_0/\kappa_C$. Fix
$$
\theta_0=\min\left\{1,\frac{\kappa_C}{8a_0},
                \frac{\kappa_C(1-r_8)}{2r_8a_0}\right\}>0,
\quad \bar a=\theta_0a_0,\quad\bar c=\theta_0c_0.
\tag{PVB.12}
$$
Use $H_8$ of {prf:ref}`lem-pvb-eighth-moment`,
$H_4=\sqrt{H_8}$, and $\bar\nu$ of
{prf:ref}`lem-pvb-joint-scores`. For $p=1,5$ put
$$
H_{{\rm p},p}=\left[((1+\bar c)H_8^{p/8})^{1/p}
                               +\sigma_JG_p^{1/p}\right]^p,
\quad M_2=cV_c+ctH_{{\rm p},1}+qG_1.
$$
Compute $B_5^{\rm src}$ in (PVB.1) with entering
$M_5=H_8^{5/8}$, $a_*=\bar a$, $c_*=\bar c$,
$H_b=\theta_0J_b$, and both gate derivative bounds replaced by $G_0$.
Compute $M_*,S_x,S_w$ in (PVB.3) with $\nu=\bar\nu$ and
$M_{{\rm p},5}=H_{{\rm p},5}$. These values bound every smaller
positive $\nu,\theta$ uniformly. In (PVB.6)--(PVB.8) use these scores
and $\bar\nu$ in every increasing denominator or factor to obtain
a fixed $\bar C_{\rm fb}<\infty$.

Use $B_4$ of {prf:ref}`thm-pvb-frozen-weighted-contraction` and put
$$
\bar B=1-r_4+r_4\bar cH_4+B_4,\quad
R=2+\frac{4\bar B}{1-r_4},\quad R_x=(R-1)^{1/4}.
$$
Compute the minorization floor $\bar\epsilon>0$ in (PVB.9) using
$\bar a,\bar c,\bar\nu,M_2,R_x$ and any fixed $u,r>0$.
Set
$$
\beta=\frac{\bar\epsilon}{r_4R+2\bar B},\quad
q_H=\max\left\{
\frac{2+\beta(r_4R+2\bar B)}{2+\beta R},
                         1-\bar\epsilon/2\right\}<1.
\tag{PVB.13}
$$
Let $L_J$ be the complete explicit primitive constant in
{prf:ref}`thm-kuhw-frozen-provider-fourth-feedback`, evaluated with this
$H_8$. Set
$$
C_J=1+\bar c(H_8^{1/8}+\sigma_JG_4^{1/4})^4,
\quad C_K=1+B_4,\quad Z_\beta=(1+\beta)/\beta,
$$
$$
\boxed{
\begin{aligned}
\theta_*&=\min\left\{\theta_0/2,
                    \frac{1-q_H}{4Z_\beta C_KL_J}\right\}>0,\\
\nu_*&=\min\left\{\bar\nu,
 \frac{1-q_H}{4Z_\beta\bar C_{\rm fb}(C_J+\theta_0L_J)}\right\}>0,\\
r_*&=(1+q_H)/2<1.
\end{aligned}}
\tag{PVB.14}
$$
All constants are functions of the stated primitive parameters.
:::

:::{prf:theorem} Positive-viscosity active population contraction and stationary law
:label: thm-pvb-active-population-convergence

Use {prf:ref}`def-pvb-record` and
{prf:ref}`def-pvb-positive-envelopes`, with
$0<\theta\le\theta_*$, $0<\nu\le\nu_*$, and the actual positive
powers $p_b=\theta\bar p_b$, $\bar p_r,\bar p_s>0$.
Let $\mathfrak G_{8,s}$ be the probability laws $\mu$ for which
$$
(x,v)\overset{\rm law}= (Y+sZ,v),\qquad
Z\sim N(0,I_d)\text{ independent of }(Y,v),\qquad
|v|\le V,\quad \mu|x|^8\le H_8.
$$
With $w_\beta=1+\beta(1+|x|^4)$, the actual complete conservative
population map satisfies
$$
\boxed{\quad
\|\mathcal F_{\nu,\theta}\mu-
                  \mathcal F_{\nu,\theta}\mu'\|_{w_\beta}
\le r_*\|\mu-\mu'\|_{w_\beta}
\quad}\qquad(\mu,\mu'\in\mathfrak G_{8,s}).
\tag{PVB.15}
$$
This class is complete and invariant. There is a unique stationary
population law $\pi_{\nu,\theta}$ in it, and
$$
\|\mu_n-\pi_{\nu,\theta}\|_{w_\beta}
\le r_*^n\|\mu_0-\pi_{\nu,\theta}\|_{w_\beta}.
\tag{PVB.16}
$$
Every invariant population law with capped velocity and finite eighth
positional moment belongs to this class and equals $\pi_{\nu,\theta}$.
For any initial law with capped velocity and finite eighth moment $M_{8,0}$,
the actual iterates enter $\mathfrak G_{8,s}$ at any integer $n_0\ge1$
satisfying
$$
\lambda_{\rm burn}^{\,n_0}M_{8,0}\le H_8/3,
\qquad \lambda_{\rm burn}=(1+3r_8)/4<1.
\tag{PVB.17}
$$
The rates and constants are population constants; they contain no particle
number. This theorem concerns the conservative population law, rather than
an invariant measure of the finite-particle interacting chain or a killed
chain's QSD.
:::

:::{prf:proof}
The tagged-root provider theorem gives, for a common root law $\mu'$
in the stated moment class,
$$
\|\mu'J_\mu-\mu'J_{\mu'}\|_{W_4}
\le\theta L_J\|\mu-\mu'\|_{W_4}.
\tag{PVB.18}
$$
Its hypotheses hold: the reward is bounded; the actual standardizer floors
are positive; the powers have the declared linear envelopes; the eighth
moments are at most $H_8$; and $\theta\le\theta_0\le\kappa_C/(8a_0)$.
In particular $c_*\le1/8$, so its complete ordered component has its
proved finite expected size. The source is exposed before exploring the
remaining component; the weighted bound retains the donor weight and
conditional source law. No independent-component replacement is used.

The frozen preparation has
$J_\mu W_4(z)\le W_4(z)+\bar c(H_8^{1/8}+\sigma_JG_4^{1/4})^4
\le C_JW_4(z)$, by its accepted donor density bound and Minkowski.
Hence for $\lambda=\mu J_\mu$,
$\lambda'=\mu'J_{\mu'}$,
$$
\|\lambda-\lambda'\|_{W_4}
\le(C_J+\theta L_J)\|\mu-\mu'\|_{W_4}.
\tag{PVB.19}
$$
Both prepared laws have speed bound $V_c$ and moment at most
$H_{{\rm p},5}$. Their weighted BV bounds are supplied by
{prf:ref}`lem-pvb-weighted-preparation-bv`, with current moment
$M_5\le H_8^{5/8}$. The first-provider interpolation preserves all
speed and field bounds, so the weighted joint scores and viscous feedback
theorem apply. The actual second provider has first velocity moment at
most $M_2$, by its own OU identity and the prepared first-moment bound.

Freeze the full population provider of $\mu$. Its root kernel $P_\mu$
has the uniform drift and minorization used in (PVB.13), and therefore
contracts $w_\beta$ variation by $q_H$. The full-law difference splits
exactly as
$$
\mathcal F\mu-\mathcal F\mu'
=(\mu-\mu')P_\mu+\mu'(P_\mu-P_{\mu'}).
$$
In the second term, first change preparation provider while keeping
kinetic providers fixed. The fixed kinetic weighted operator is bounded
by $C_K$ from the conditional fourth-moment drift. This costs at most
$C_K\theta L_J\|\mu-\mu'\|_{W_4}$.
Then hold the own prepared law $\lambda'$ fixed and change its two
kinetic providers. By (PVB.8) and (PVB.19), this costs at most
$\nu\bar C_{\rm fb}(C_J+\theta L_J)
\|\mu-\mu'\|_{W_4}$.
Since
$\beta W_4\le w_\beta\le(1+\beta)W_4$, the total coefficient is
$$
q_H+Z_\beta[C_K\theta L_J+
                  \nu\bar C_{\rm fb}(C_J+\theta L_J)]\le r_*.
$$
The two positive endpoints (PVB.14) prove that last inequality and (PVB.15).

The weighted finite-measure variation space is Banach: multiplication
of a signed measure by the positive function $w_\beta$ identifies it
isometrically with the finite-measure variation space. Probabilities are
closed in this norm. The velocity cap and eighth-moment sublevel are
closed under weak convergence and hence under this norm convergence.
For the Gaussian-mixture condition, choose representing latent laws
$\eta_j=\operatorname{Law}(Y_j,v_j)$. Jensen conditional on $Y_j$ gives
$\eta_j|Y_j|^8\le\mu_j|x|^8\le H_8$, so these latent laws are tight.
Along a weakly convergent subsequence, their Gaussian convolutions
converge weakly to the convolution of the latent limit. It must equal
the weighted-variation limit $\mu$. Thus $\mathfrak G_{8,s}$ is closed
and complete. It is nonempty: $\operatorname{Law}(sZ,0)$ has eighth
moment at most $H_8$ by the definition of $B_8,H_8$.

The moment lemma and the independent final position noise make this class
invariant. Banach's contraction argument gives a unique fixed law there
and (PVB.16). Since $\theta\le\theta_0/2$,
$r_8(1+c_*)\le(1+3r_8)/4=\lambda_{\rm burn}$.
The actual moment recursion therefore gives
$$
M_{8,n}\le\lambda_{\rm burn}^{\,n}M_{8,0}+2H_8/3.
$$
After any positive number of updates the Gaussian-mixture representation
holds. Condition (PVB.17) proves entry into the class. An invariant law
of finite eighth moment satisfies the same moment recursion, hence has
moment at most $2H_8/3$ and the same representation. Its uniqueness
follows from the contraction already proved.
:::

:::{prf:corollary} Physical Wasserstein relaxation of the alive population law
:label: cor-pvb-alive-w2-relaxation

Under {prf:ref}`thm-pvb-active-population-convergence`, let $G$ be any
positive physical phase-space matrix. For $\mu_0\in\mathfrak G_{8,s}$,
$$
W_{2,G}(\mu_n,\pi_{\nu,\theta})^2
\le\frac{2\lambda_{\max}(G)(1+V^2)}{\beta}
 r_*^n\|\mu_0-\pi_{\nu,\theta}\|_{w_\beta}.
\tag{PVB.20}
$$
In particular the initial norm is at most $2+2\beta(1+\sqrt{H_8})$.
For other finite-eighth-moment input laws the bound applies after the
finite burn-in (PVB.17). In physical time the $W_2$ exponential rate is
$-\log(r_*)/(2h)$ and the weighted variation rate is $-\log(r_*)/h$.
:::

:::{prf:proof}
Match the common part of two phase laws identically and couple their
remaining parts. The physical squared cost is at most
$2\lambda_{\max}(G)\int(|x|^2+|v|^2)|\mu-\mu'|$.
Since $|x|^2+|v|^2\le(1+V^2)W_4(x)$ and
$\beta W_4\le w_\beta$, this proves the displayed transport bound
in terms of the weighted variation. Substitute (PVB.16).
The sum of the two $w_\beta$ moments is at most the asserted initial
constant by Cauchy--Schwarz and the eighth-moment bound. Taking square
roots yields the stated physical-time rate.
:::

:::{prf:corollary} Nonempty primitive regime at the original harmonic step size
:label: cor-pvb-original-step-active-viscosity

Take $d=3$, $h=0.04$, $\gamma=b_O=1$, $\sigma_x=0.1$,
$V=2$, $\rho=1$, $\alpha_{\rm col}=1/2$, and $\sigma_J=0.1$.
Retain the actual $F=-x$ and configure the bounded reward
$R(x)=-\tanh(|x|^2/2)$, so $R_{\rm osc}=1$ and $L_R\le2$.
Use any positive declared comparison radii, companion widths,
logistic amplitudes/base floors, standardizer floors, acceptance
regularizer and scale, and positive reference powers.
Then the exact primitive formulas above give $\theta_*,\nu_*>0$.
The complete active population law converges as proved for every
$0<\theta\le\theta_*$ and $0<\nu\le\nu_*$.

This is a sufficient regime with the declared bounded reward channel.
It does not certify the default raw quadratic reward or the reference
$\nu=0.3$, and it excludes no exact or near fitness ties.
:::

:::{prf:proof}
Here $t=0.02$, $0<c<1$, $m=1-0.02^2>0$, and
$a_x=1-0.02^2(1+c)\in(0,1)$. Both Gaussian variances are positive,
$V_c=4$, and every displayed drift, floor, score and provider constant
is finite. The fourth- and eighth-moment drift coefficients are strictly
below one. The local Gaussian floor and $\beta$ are strictly positive,
so $q_H<1$. The two feedback constants are finite, making both endpoints
(PVB.14) strictly positive. For the reward derivative,
$|\nabla R|=|x|\operatorname{sech}^2(|x|^2/2)\le2$ when $|x|\le2$;
for $|x|\ge2$, it is at most $4|x|e^{-|x|^2}\le8e^{-4}<2$.
The selection powers are positive throughout the interval. Unequal
fitness inputs can have accepted copying, while the gate's continuous
zero branch covers exact ties.
:::

The remaining particle-level task is a uniform-in-time transfer from the
actual finite-swarm law to this population law using its actual one-update
consistency estimate. Killed-law conditioning requires an additional
whole-array survival comparison. Neither conclusion is inferred from the
population theorem alone.
