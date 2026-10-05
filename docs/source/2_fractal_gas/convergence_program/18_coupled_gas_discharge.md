# Coupled Viscous Gas: Kinetic Estimates and Finite-Population Discharge

(sec-coupled-gas-register)=
## 1. Complete parameter record and law

:::{div} feynman-prose
Take the viscous gas that the program already runs and follow one complete
update. The neighbors now influence each velocity kick, so a proof that treated
the walkers independently needs a fresh calculation at those two kicks. The
donor draws, sampled fitness, component collisions, Gaussian innovations and
terminal check keep their declared meanings. Our parameter record lets you
inspect every one of those choices.

The equilibrium question also needs a precise object. A swarm can become
extinct; the quasi-stationary law describes its distribution conditional on
still having at least one living row. Retained dead coordinates stay in that
distribution because the next update reads them. Later, we use a change of
probability weights to analyze this law. Keeping track of those weights is
what lets the final estimates describe the actual surviving gas.
:::

:::{prf:definition} Coupled gas parameter record and recording extension
:label: def-cgd-parameter-register

Take the complete canonical record $\theta$ of
{prf:ref}`def-slc-parameter-register`, with its actual donor conventions,
sampled fitness, gates, connected-component collision, retained dead coordinates,
terminal classification, and no historical donor pool. Extend it by

$$
\Theta=(\theta;\nu,\rho,\mathfrak n,\Gamma,\vartheta_\Gamma),
\qquad \nu\ge0,\quad\rho>0,
\qquad\mathfrak n\in\{\mathrm{count},\mathrm{row}\}.
$$

The two B stages are exactly those of
{prf:ref}`def-variant-viscous-euclidean`. The Gaussian kernel uses the physical
positions and the declared bandwidth $\rho$. Every row is eligible at both
kicks because revival precedes kinetics. $\Gamma$ is a specified measurable
geometry/Fractal Set readout, and $\vartheta_\Gamma$ records all its projection,
ridge, spectral-floor, neighbor, tie-breaking and empty-pool conventions.
Recording $\Gamma$ does not change the reward, donor probabilities, force or
noise. A configuration using its output as feedback requires a separate
kernel theorem.

In this chapter set

$$
t=h/2,\quad c=e^{-\gamma h},\quad b=t(1+c),\quad
q^2=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,
\end{cases}
\quad s^2=\sigma_x^2h,
$$

$$
V=V_{\max},\qquad V_c=(1+2|\alpha_{\rm col}|)V,
\qquad \ell_K=e^{-1/2}/\rho.
$$

The letters $t,c$ here are kinetic coefficients, not physical coordinates or
the collision parameter. The independent innovations are standard Gaussian
vectors. For a population $z=(z_i)_{i=1}^N$, write
$\|z\|_{2,N}^2=N^{-1}\sum_i|z_i|^2$ and
$\|z\|_{\infty,N}=\max_i|z_i|$.

The finite-population law considered below is the QSD of the *full killed
marked kernel*, not a Gibbs comparison, a conservative invariant law or a
one-particle mean-field stationary law. Its Doob transform has its own
invariant law. The unbounded-domain moment estimates in Section 3 are
one-step estimates and do not assert a QSD on an unbounded domain.
:::

:::{prf:definition} Existing viscous reference parameter instance
:label: def-cgd-existing-reference

The reference instance is the unchanged `RunConfig::viscous_euclidean()` of
{prf:ref}`rem-variant-viscous-euclidean-rust`, together with its unchanged
`GasConfig::euclidean` components in {prf:ref}`rem-variant-euclidean-rust`:

$$
\begin{gathered}
d=3,\quad N=200,\quad h=0.04,\quad \gamma=1,\quad b_O=1,\quad
\sigma_x=0.1,\quad\sigma_J=0.1,\quad V=2,\quad\alpha_{\rm col}=0.5,\\
\nu=0.3,\quad\rho=1,\quad\mathfrak n=\mathrm{count},\quad
U(x)=|x|^2/2,\quad R(x,v)=-U(x),\quad D=[-2,2]^3,\\
R_x^{\rm feat}=R_v^{\rm feat}=2,\quad\lambda_{\rm alg}=1,\quad
\epsilon_D=\epsilon_C=2,\quad\delta_D=10^{-3},\\
A_r=A_s=2,\quad\eta_r=\eta_s=0.1,\quad p_r=p_s=1,\quad
\sigma_r=\sigma_s=0.1,\quad s_c=1,\quad\epsilon_c=10^{-6}.
\end{gathered}
$$

Both companion roles use independent Gaussian draws with no self companion
except the declared singleton convention; gates occur every update. The
collision uses one Haar matrix per accepted connected component, the
boundary is classified only at the end, all dead coordinates are retained,
and no historical donors or geometry feedback are used. There is no curl
rotation. The configured variant has $\Gamma=\varnothing$; a declared passive
record readout is covered by {prf:ref}`thm-cgd-readout-law`.
:::

:::{prf:remark} Real-coordinate kernel and numerical execution
:label: rem-cgd-real-kernel-scope

The analytic kernel in this chapter is the real-arithmetic, independent
Gaussian-innovation transition already defined in
{prf:ref}`def-eg-baoab-canonical` and
{prf:ref}`def-variant-viscous-euclidean`. Absolute continuity, analytic Gaussian
weights, full support, QSD and tail assertions refer to that existing
mathematical kernel. A recorded finite-precision execution has its declared
arithmetic and innovation representation; its bit-level state law is not
asserted to have a Lebesgue density. Numerical rounding, overflow and
underflow effects are separate comparison errors, including any implementation
branch taken when a computed row normalizer is zero. No new noise law or
algorithmic update is introduced by this scope distinction.
:::

(sec-coupled-gas-alignment)=
## 2. Exact alignment estimates for both normalizations

:::{div} feynman-prose
Imagine two walkers pulling their velocities toward one another. With the
same coefficient in both directions, one loses exactly the momentum the other
gains. Add all those pairs and the average velocity stays fixed while the
differences shrink. Count normalization preserves this symmetry, and the
matrix calculation below shows that an appropriately sized kick cannot
increase the population's mean kinetic energy.

Row normalization divides each walker's pull by its own total neighbor weight.
It still makes every outgoing velocity a convex average. The natural energy
weight then becomes that walker's degree, the sum of its neighbor weights.
To recover an ordinary population average we pay the largest-to-smallest
degree ratio. Writing that price explicitly matters when a sparse neighborhood
and a dense neighborhood coexist.
:::

:::{prf:lemma} Count-normalized graph Laplacian and viscous kick
:label: lem-cgd-count-kick

For fixed positions define $K_{ij}=K_\rho(x_i,x_j)$ for $i\ne j$ and

$$
(L_xv)_i=\frac1N\sum_{j\ne i}K_{ij}(v_i-v_j).
$$

Then $L_x$ is symmetric, $L_x\mathbf1=0$, and
$0\preceq L_x\preceq I$. Consequently

$$
\|F^{\rm visc}(x,v)\|_{2,N}\le\nu\|v\|_{2,N},
\qquad \|F^{\rm visc}(x,v)\|_{\infty,N}
\le2\nu\|v\|_{\infty,N}.
$$

If $0\le t\nu\le1$, $W_x=I-t\nu L_x$ is a nonnegative symmetric
doubly stochastic matrix, and, for every $p\ge1$,

$$
\sum_i|(W_xv)_i|^p\le\sum_i|v_i|^p,\qquad
\|W_xv\|_{\infty,N}\le\|v\|_{\infty,N},\qquad
\sum_i(W_xv)_i=\sum_iv_i.
$$

The force-work identity is

$$
\frac1N\sum_i v_i\cdot F_i^{\rm visc}
=-\frac{\nu}{2N^2}\sum_{i,j}K_{ij}|v_i-v_j|^2.
$$
:::

:::{prf:proof}
For every scalar vector $u$,

$$
u^{\mathsf T}L_xu=\frac1{2N}\sum_{i,j}K_{ij}(u_i-u_j)^2
\le\frac1{2N}\sum_{i,j}(u_i-u_j)^2
=\sum_i(u_i-\bar u)^2\le|u|^2.
$$

This proves both operator inequalities, separately in every velocity
coordinate. Since $F^{\rm visc}=-\nu L_xv$, it proves the mean-square force
bound. The rowwise bound follows from $K_{ij}\le1$ and the triangle inequality.
The diagonal of $W_x$ is $1-t\nu N^{-1}\sum_{j\ne i}K_{ij}\ge0$; its
off-diagonal entries are $t\nu K_{ij}/N$, and its row and column sums are one.
Convexity of $|\cdot|^p$ gives
$|(W_xv)_i|^p\le\sum_j(W_x)_{ij}|v_j|^p$.
Sum over $i$ using the column sums. The maximum and momentum identities follow
from the row and column sums. Finally pair the terms $(i,j)$ and $(j,i)$ in
the force-work sum. $\square$
:::

:::{prf:lemma} Row-normalized alignment, degree measure and comparison cost
:label: lem-cgd-row-kick

For $N\ge2$ let $d_i=\sum_{j\ne i}K_{ij}>0$ and
$\omega_{ij}=K_{ij}/d_i$. For $0\le t\nu\le1$, the viscous kick is the
row-stochastic matrix

$$
P_x=(1-t\nu)I+t\nu\omega.
$$

For every $p\ge1$,

$$
\|P_xv\|_{\infty,N}\le\|v\|_{\infty,N},\qquad
\sum_i d_i|(P_xv)_i|^p\le\sum_i d_i|v_i|^p.
$$

Writing $\chi(x)=\max_i d_i/\min_i d_i$, this implies

$$
\frac1N\sum_i|(P_xv)_i|^p
\le\chi(x)\frac1N\sum_i|v_i|^p,
$$

and the exact force-work identity is

$$
\sum_i d_i v_i\cdot F_i^{\rm visc}
=-\frac\nu2\sum_{i,j}K_{ij}|v_i-v_j|^2.
$$

If $\max_i|x_i|\le R$, then
$\chi(x)\le e^{2R^2/\rho^2}$. No such bound with a fixed finite $R$ is
asserted for positions after an unbounded Gaussian drift. For $N=1$ the
force is zero and the kick is the identity.
:::

:::{prf:proof}
The entries of $P_x$ are nonnegative and every row sums to one. Moreover
$d_i\omega_{ij}=K_{ij}=d_j\omega_{ji}$, so
$\sum_i d_i(P_x)_{ij}=d_j$. Apply Jensen's inequality to each row and sum with
weights $d_i$. Comparison of the largest and smallest weights proves the
unweighted estimate. Pairing terms proves the work identity. On the displayed
position ball, $e^{-2R^2/\rho^2}\le K_{ij}\le1$, hence
$(N-1)e^{-2R^2/\rho^2}\le d_i\le N-1$. $\square$
:::

:::{prf:lemma} Position dependence of the viscous force
:label: lem-cgd-viscous-local-lipschitz

Let $\delta_x=\|x-y\|_{\infty,N}$,
$\delta_v=\|v-w\|_{\infty,N}$ and
$\|w\|_{\infty,N}\le M$. For count normalization,

$$
\|F^{\rm visc}(x,v)-F^{\rm visc}(y,w)\|_{\infty,N}
\le2\nu\delta_v+4\nu M\ell_K\delta_x.
$$

For row normalization, when all positions in both populations have norm at
most $R$, the same left side is at most

$$
2\nu\delta_v+4\nu M\ell_K e^{2R^2/\rho^2}\delta_x.
$$

These are bounds on physical coordinates. Bounded feature distance alone does
not supply a global physical Lipschitz bound.
:::

:::{prf:proof}
The Gaussian kernel has gradient norm at most $\ell_K$ in either position
argument, so $|K_{ij}(x)-K_{ij}(y)|\le2\ell_K\delta_x$.
For count normalization split the force difference into the velocity
difference and the kernel difference. Their rowwise bounds are $2\nu\delta_v$
and $N^{-1}\nu\sum_{j\ne i}(2\ell_K\delta_x)(2M)$.
For row normalization, put $k_*=e^{-2R^2/\rho^2}$ and use

$$
\sum_{j\ne i}|\omega_{ij}(x)-\omega_{ij}(y)|
\le\frac{2\sum_{j\ne i}|K_{ij}(x)-K_{ij}(y)|}{d_i(x)}
\le\frac{4\ell_K\delta_x}{k_*}.
$$

The row probabilities sum to one, so their difference sums to zero. Thus the
kernel-difference term can be written as
$\nu\sum_j[\omega_{ij}(x)-\omega_{ij}(y)]w_j$ and bounded by the displayed
quantity times $\nu M$. $\square$
:::

(sec-coupled-gas-moments)=
## 3. Uncapped intermediate moments and terminal survival

:::{div} feynman-prose
The velocity cap acts at the end. Before then, the OU innovation can produce
an arbitrarily large velocity, and the second viscous kick reads it. We must
therefore estimate that intermediate velocity rather than replace it by the
cap radius.

For count normalization the answer comes from the contraction just proved.
Apply it to the entire realized population after the innovation: it bounds the
mean square even though the second set of weights depends on those same
velocities through the drifted positions. This avoids paying for the largest
Gaussian draw among all walkers. Positions offer another exact simplification.
The second kick changes velocity only, so the final position retains an
explicit Gaussian law conditional on the entering population. That law gives
the terminal death probabilities directly.
:::

:::{prf:theorem} Exact coupled position law and population-independent moments
:label: thm-cgd-kinetic-moments

Use count normalization and $t\nu\le1$. Suppose the actual acceleration is
globally Lipschitz with $|F(x)|\le B_F+L_F|x|$. For the whole post-collision,
post-jitter population put $X=\|x\|_{2,N}$, $U=\|v\|_{2,N}$ and
$A=U+t(B_F+L_FX)$. The uncapped stages satisfy

$$
\|v_1\|_{2,N}\le A,\qquad
x_2=x+bv_1+tq\xi_v,
\qquad x^+=x+bv_1+tq\xi_v+s\xi_x.
$$

Conditional on the entering population, the final positions are independent
Gaussians with means $M_i=x_i+bv_{1,i}$ and common covariance
$s_h^2I_d$, where $s_h^2=t^2q^2+s^2$. In particular,

$$
\begin{aligned}
\mathbb E\|v_2\|_{2,N}^2&\le c^2A^2+dq^2,\\
\mathbb E\|x_2\|_{2,N}^2&\le(X+bA)^2+dt^2q^2,\\
\mathbb E\|x^+\|_{2,N}^2&\le(X+bA)^2+ds_h^2,\\
\mathbb E\|v_3\|_{2,N}^2
&\le2(c^2A^2+dq^2)
 +4t^2\{B_F^2+L_F^2[(X+bA)^2+dt^2q^2]\},\\
\mathbb E\|x^+-x\|_{2,N}^2
&\le3b^2[U^2+t^2B_F^2+t^2L_F^2X^2]+ds_h^2.
\end{aligned}
$$

The velocity cap gives $\|v^+\|_{2,N}^2\le V^2$ and
$\mathbb E\|v^+-v\|_{2,N}^2\le2V^2+2U^2$.
Every constant in these estimates is independent of $N$. For an exchangeable
random entering population, taking expectation gives the corresponding tagged
second moments.
:::

:::{prf:proof}
The first kick is $v_1=W_xv+tF(x)$. Apply
{prf:ref}`lem-cgd-count-kick`, then the triangle inequality in the normalized
Euclidean norm and $\|F(x)\|_{2,N}\le B_F+L_FX$.
Substitute $v_2=cv_1+q\xi_v$ into the two drifts to obtain the exact position
identities. The B2 kick and cap occur later and leave positions unchanged.
The two independent Gaussian terms give the stated conditional laws and
second moments. Although $W_{x_2}$ depends on the realized innovations, its
contraction is pathwise, so

$$
|\!|v_3|\!|_{2,N}^2
\le2|\!|v_2|\!|_{2,N}^2+2t^2|\!|F(x_2)|\!|_{2,N}^2
\le2|\!|v_2|\!|_{2,N}^2+4t^2(B_F^2+L_F^2|\!|x_2|\!|_{2,N}^2).
$$

This handles the uncapped velocities at the second force evaluation without a
maximum-over-$N$ estimate. Expand the Gaussian position increment and use
$(a_1+a_2+a_3)^2\le3(a_1^2+a_2^2+a_3^2)$ to obtain the increment bound.
The cap estimates follow directly from its radius. Exchangeability makes the
expected mean of the row moments equal to every tagged moment. $\square$
:::

:::{div} feynman-prose
The useful cancellation happens before taking expectations. Even a very large
OU draw is still passed through an alignment matrix that contracts the whole
population's mean square. That is why the constants above contain the
dimension and noise amplitudes but no factor growing with the number of
walkers. These are one-update bounds conditional on the entering moments.
They supply the intermediate estimates needed by a longer argument; they do
not by themselves prove confinement for unlimited time.
:::

:::{prf:corollary} Coupled death-probability comparison
:label: cor-cgd-boundary-comparison

For either normalization, conditional on an entering population,

$$
\mathbb P\{a_i^+=0\}=1-\int_D
\frac{e^{-|z-M_i|^2/(2s_h^2)}}{(2\pi s_h^2)^{d/2}}\,dz.
$$

If $s_h>0$, two entering populations with velocity norm at most $M$ satisfy,
for count normalization,

$$
|p_{{\rm dead},i}(x,v)-p_{{\rm dead},i}(y,w)|
\le\frac{[1+bt(L_F+4\nu M\ell_K)]\delta_x
 +b(1+2t\nu)\delta_v}{\sqrt{2\pi}s_h}.
$$

For row normalization on the position ball of radius $R$, replace $\ell_K$
in its viscous term by $\ell_K e^{2R^2/\rho^2}$. A Lebesgue-null terminal
boundary has zero landing probability. No independence of final velocities is
asserted.
:::

:::{prf:proof}
The exact position identity does not use any property of B2. Apply
{prf:ref}`lem-cgd-viscous-local-lipschitz` to
$M=x+b[v+tF(x)+tF^{\rm visc}(x,v)]$. Two Gaussians of covariance
$s_h^2I$ and mean separation $r$ differ in event probabilities by at most
$r/(\sqrt{2\pi}s_h)$, as calculated in
{prf:ref}`lem-euclidean-boundary-holder`. Absolute continuity proves the last
assertion. $\square$
:::

(sec-coupled-gas-phase-smoothing)=
## 4. Nonlinear joint phase-space smoothing

:::{div} feynman-prose
Knowing that every final position has a Gaussian density is useful, but the
equilibrium law also contains velocities. Their second kick reads the whole
drifted swarm, so the joint output no longer has the simple affine Gaussian
formula used for zero viscosity.

For the quadratic reference force, we can still write the second kick as one
explicit map of the OU output. The positive margin below makes that map grow
away from the origin: it cannot hide an arbitrarily large input inside a
bounded output. A degree argument shows that it reaches every velocity target.
Its analytic derivative fails to be invertible only on a set of zero Gaussian
probability. Those two facts recover a density, full support, and continuity
of the whole output law. They account for the coupling without assuming
independent final velocities.
:::

:::{prf:lemma} Continuity in total variation under locally nonsingular maps
:label: lem-cgd-pushforward-tv

Let $f_n,f:\mathbb R^m\to\mathbb R^m$ be $C^1$ maps such that $f_n\to f$
in $C^1$ on every compact set. Let $Z$ have an integrable continuous density,
and suppose $\det Df(Z)\ne0$ almost surely. Then
$\|\mathcal L(f_n(Z))-\mathcal L(f(Z))\|_{\rm TV}\to0$.
:::

:::{prf:proof}
Discard a tail and a neighborhood of the critical set with total probability
less than an arbitrary $\varepsilon>0$. The inverse function theorem covers
the remaining compact set by finitely many open sets on each of which $f$ is
a diffeomorphism. Partition the retained source measure into finitely many
measures supported on smaller compact pieces inside these sets. The pieces
can be chosen with boundary of Lebesgue measure zero: use sufficiently small
boxes and successively subtract their preceding overlaps. Discard any residual
measure of mass less than $\varepsilon$.

On a slightly larger neighborhood of each piece, the inverse function theorem
and $C^1$ convergence make $f_n$ a diffeomorphism for all sufficiently large
$n$, with inverse and inverse Jacobian converging uniformly on compact
subsets of the limiting image. The change-of-variables density is
$g(f_n^{-1}(y))|\det Df_n^{-1}(y)|$ times the indicator of the source piece.
The piece boundaries have null images because the maps are locally Lipschitz.
Hence these densities converge almost everywhere, and their integrals equal
the fixed mass of the source piece. Scheffé's elementary identity
$\int|p_n-p|=\int p_n+\int p-2\int\min(p_n,p)$ gives $L^1$ convergence.
Sum over the finitely many pieces. The discarded source mass contributes at
most $4\varepsilon$ to the total-variation norm convention
$\sup_{|g|\le1}|\mu g-\nu g|$. Let $\varepsilon\downarrow0$. $\square$
:::

:::{prf:theorem} Coupled quadratic B2 map: surjectivity and smoothing
:label: thm-cgd-phase-smoothing

Take $F(x)=-\lambda x$, $\lambda\ge0$, $q,s>0$, and define

$$
\alpha=1-\lambda t^2,\qquad
\kappa=\begin{cases}\alpha-t\nu,&\mathfrak n=\mathrm{count},\\
\alpha-2t\nu,&\mathfrak n=\mathrm{row}.
\end{cases}
\tag{CGD.1}
$$

Assume $\kappa>0$. For a fixed A1 population $x_1$, the actual B2 map of the
OU output $z=v_2$ is

$$
T_{x_1}(z)=\alpha z-t\lambda x_1
+tF^{\rm visc}(x_1+tz,z).
\tag{CGD.2}
$$

It is real analytic, proper and surjective. Its critical set has Lebesgue
measure zero. Conditional on any fixed collision pattern, shared rotations
and cloning jitters, the joint pre-cap output $(x^+,v_3)$ has an absolutely
continuous law with full support on $\mathbb R^{2dN}$, and that law varies
continuously in total variation with the entering coordinates. The capped
output has an absolutely continuous law with full support on
$\mathbb R^{dN}\times B_V^N$.
:::

:::{prf:proof}
**1. Coercivity and surjectivity.** For count normalization use the ordinary
product Euclidean norm and {prf:ref}`lem-cgd-count-kick`; for row normalization
use the maximum row norm and its force bound. In the corresponding norm,

$$
\|T_{x_1}(z)\|\ge\kappa\|z\|-t\lambda\|x_1\|.
\tag{CGD.3}
$$

Thus inverse images of compact sets are bounded and closed. For a proposed
target $y$, consider
$H_u(z)=\alpha z-t\lambda x_1+utF^{\rm visc}(x_1+tz,z)-y$,
$0\le u\le1$. Inequality (CGD.3) with the additional subtraction $y$ shows
that no $H_u$ vanishes on the boundary of a sufficiently large ball. The
Brouwer degree therefore equals that of the invertible affine map $H_0$,
which is one. A nonzero degree gives a zero of $H_1$. This proves surjectivity.

**2. The critical set is null.** Gaussian kernels and the positive row
denominators make $T_{x_1}$ real analytic. At a consensus velocity
$z_i=z_0$ the terms differentiating the position dependence of the weights
vanish, since every difference $z_j-z_i$ is zero. Hence

$$
DT_{x_1}(z)=\alpha I-t\nu L_{x_1+tz_0}
$$

with the count or row Laplacian. In the count norm its Laplacian norm is at
most one; in the maximum row norm it is at most two. Condition (CGD.1) makes
this derivative invertible by its convergent Neumann series. Its determinant
is consequently a nonzero real analytic function. A nontrivial real analytic
function on a connected open set has a null zero set: this follows locally by
induction on dimension, taking the first nonzero coefficient in a one-variable
power series and applying Fubini; a countable cover gives the global assertion.

**3. Joint smoothing.** Here $z=cv_1+q\xi_v$ has a strictly positive Gaussian
density, and
$x^+=x_1+tz+s\xi_x$. The map from $(z,\xi_x)$ to
$(x^+,T_{x_1}(z))$ has determinant, up to sign,
$s^{dN}\det DT_{x_1}(z)$, which is nonzero almost everywhere. Cover its
noncritical set by countably many inverse-function charts. Change of variables
on those charts shows that the pushforward is absolutely continuous; the
critical source set carries zero Gaussian mass. Surjectivity of $T_{x_1}$
and free choice of $\xi_x$ make the joint map surjective. The inverse image of
any nonempty open output set is open and nonempty, and therefore has positive
Gaussian mass. This proves full support.

For converging entering coordinates the map converges in $C^1$ on compact
sets. Apply {prf:ref}`lem-cgd-pushforward-tv`, representing $z$ through its
fixed standard Gaussian innovation. Finally the radial cap is a $C^1$
diffeomorphism from $\mathbb R^d$ onto $B_V$, with positive Jacobian.
It preserves absolute continuity and full support and contracts total
variation. $\square$
:::

(sec-coupled-gas-qsd)=
## 5. Explicit finite-population minorization and QSD

:::{div} feynman-prose
We now need a piece of the transition law that every starting swarm shares.
Pick a small target region inside the existing terminal box and velocity ball.
There is a positive probability that the cloning jitters and OU draws land
in convenient regions, after which the final position noise can reach that
target. The next formulas give a common lower bound for this event.

The Gaussian noises remain unbounded in the algorithm. Selecting a favorable
event is just how we prove a lower bound. Likewise, the terminal box and
velocity cap are already configured components of this variant. The spectral
argument then gives its quasi-stationary law. For a fixed population size the
common part is positive; as the population grows its displayed bound can
become very small. The proof keeps that dependence visible.
:::

:::{prf:definition} Primitive-parameter minorization certificate
:label: def-cgd-minorization-certificate

Assume the actual terminal box $D=[-L_D,L_D]^d$, $L_D>0$, the quadratic force of
{prf:ref}`thm-cgd-phase-smoothing`, and all other canonical parameters in
$\Theta$. Choose deterministic analysis radii $J,G>0$, $0<L_0<L_D$ and
$0<r_v<V$. These restrict events in a proof; they do not truncate the noises
or alter the algorithm. Put $m=dN$, $R_D=\sqrt d L_D$,
$R_0=\sqrt d L_0$, and

$$
\begin{aligned}
p_J&=\begin{cases}G_d(J/\sigma_J),&\sigma_J>0,\\1,&\sigma_J=0,\end{cases}
&p_G&=G_d(G),\\
B_0&=R_D+J,&B_1&=(1+2t\nu)V_c+t\lambda B_0,
&B_x&=B_0+tB_1,\\
Y&=\frac{Vr_v}{V-r_v},
&Z&=\begin{cases}\sqrt N(Y+t\lambda B_x)/\kappa,&\mathfrak n=\mathrm{count},\\
(Y+t\lambda B_x)/\kappa,&\mathfrak n=\mathrm{row},\end{cases}
&R_2&=B_x+tZ.
\end{aligned}
$$

Here $G_d$ is the explicit Gaussian-ball probability in
{prf:ref}`def-slc-parameter-register`. For $N=1$ set $D_T=\alpha$. For
$N\ge2$ set

$$
D_T=\begin{cases}
\alpha+t(2\nu+4t\nu\ell_K Z),&\mathfrak n=\mathrm{count},\\
\alpha+t(2\nu+8t\nu\ell_K Z e^{2R_2^2/\rho^2}),&\mathfrak n=\mathrm{row}.
\end{cases}
$$

Define the positive numbers

$$
\begin{aligned}
g_*&=(2\pi q^2)^{-m/2}(2\pi s^2)^{-m/2}
\exp\!\left[-\frac{N(Z+cB_1)^2}{2q^2}
-\frac{N(R_0+B_x+tZ)^2}{2s^2}\right],\\
J_*&=(\sqrt N D_T)^m,
&|A_0|&=(2L_0)^m[v_d(r_v)]^N,\\
\epsilon_N&=\min\left\{\frac12,\ p_J^N\frac{g_*|A_0|}{J_*}\right\},\\
Z_0&=cB_1+qG,
&\zeta_N&=p_J^Np_G^N
\left[1-\Phi\!\left(\frac{L_D+B_x+tZ_0}{s}\right)\right]^N.
\end{aligned}
\tag{CGD.4}
$$

The set $A_0$ is the all-alive product with each position in
$[-L_0,L_0]^d$ and each velocity in $\overline B(0,r_v)$.
$\theta_0$ is normalized Lebesgue measure on this set. The symbols
$v_d,\Phi$ denote the Euclidean ball volume and standard normal distribution
function. All constants in (CGD.4) are explicit functions of the complete
record $\Theta$ and the declared analysis radii; their independence of unused
reward and donor parameters follows from the uniform argument over patterns.
:::

:::{prf:theorem} Unique coupled QSD and conditioned convergence
:label: thm-cgd-finite-n-qsd

Fix $N,d\ge1$, the terminal-box canonical marked algorithm in
{prf:ref}`def-cgd-parameter-register`, $q,s>0$, globally defined
$U(x)=\lambda|x|^2/2$, and $\kappa>0$ from (CGD.1). Keep all positive donor
widths and regularization/fitness floors required by the canonical kernel.
Then its actual coupled killed kernel $Q_N^\Theta$ has a unique QSD
$\nu_N^\Theta$ and eigenvalue $\alpha_N^\Theta\in(0,1)$, with

$$
Q_N^\Theta(k,\cdot)\ge\epsilon_N\theta_0(\cdot),\qquad
\epsilon_N\le Q_N^\Theta1(k)\le1-\zeta_N,
\qquad\epsilon_N\le\alpha_N^\Theta\le1-\zeta_N.
\tag{CGD.5}
$$

There is a positive continuous eigenfunction $e_N^\Theta$, normalized by
$\max e_N^\Theta=1$, and its exact spectral quantities

$$
m_N^\Theta=\min e_N^\Theta>0,\quad
\delta_N^\Theta=\frac{\epsilon_N\theta_0(e_N^\Theta)}{\alpha_N^\Theta}>0
$$

give, for every initial law $\eta$ on actual nonextinct capped states,

$$
\left\|\frac{\eta(Q_N^\Theta)^n}{\eta(Q_N^\Theta)^n1}
-\nu_N^\Theta\right\|_{\rm TV}
\le\frac4{(m_N^\Theta)^2}(1-\delta_N^\Theta)^n.
\tag{CGD.6}
$$

The quantities $m_N^\Theta$ and $\theta_0(e_N^\Theta)$ are defined by the
identified full-kernel eigenproblem. The primitive bounds in
{prf:ref}`thm-cgd-primitive-eigenfunction` replace them in the quantitative
subregime derived below, including the existing reference configuration.
Neither theorem asserts uniformity in $N$. Dead physical positions remain
unbounded and retained.
:::

:::{prf:proof}
**1. Compactify only the input feature coordinates.** Use exactly the
compact space $K_N$ of the proof of
{prf:ref}`thm-chaos-canonical-finite-n-qsd`: alive positions in $\overline D$,
dead positions represented by their bounded donor feature, capped velocities,
and the finite nonempty masks. The force only sees the post-revival population.
Raw dead input positions are not used after the donor draw, because revival
copies an alive donor before jitter. Viscosity introduces no new dependence
on those raw coordinates. Thus the coupled kernel extends to this compact
input representation without discarding any actual coordinate.

**2. Bound the common part of the output density.** Condition on any
companion/gate pattern and any shared rotations. The component velocity is
bounded by $V_c$. The event that all preassigned clone jitters have norm at
most $J$ has probability $p_J^N$ and gives $|x_i|\le B_0$ before B1,
$|v_{1,i}|\le B_1$ and $|x_{1,i}|\le B_x$.
For a pre-cap target $y$ with every row norm at most $Y$, coercivity (CGD.3)
and surjectivity give a preimage $z$ with $|z_i|\le Z$.

On that bounded source set differentiate (CGD.2). For count normalization,
the linear velocity derivative is bounded in the maximum row norm by $2\nu$,
and the derivative through $x_1+tz$ by $4t\nu\ell_KZ$. For row normalization,
the degree is at least $(N-1)e^{-2R_2^2/\rho^2}$; the normalized-weight
derivative estimate in {prf:ref}`lem-cgd-viscous-local-lipschitz`, or its
direct quotient-rule calculation, gives the conservative bound
$8t\nu\ell_KZ e^{2R_2^2/\rho^2}$. Hence
$\|DT\|_{2\to2}\le\sqrt N D_T$ and $|\det DT|\le J_*$.

For any fixed $x_1$, critical values of this smooth map form a null set.
For every regular target $y$, its preimage is discrete and compact, hence
finite, and has at least one member in the source bound. Change of variables
therefore gives a density at least the Gaussian density at that member
divided by $J_*$. For a simultaneous position target in $[-L_0,L_0]^{dN}$,
the OU density and the conditional final position Gaussian density are bounded
below by $g_*$. The cap inverse has determinant at least one: its forward
radial and tangential derivatives are at most one. Thus the capped density
is at least $g_*/J_*$ almost everywhere on $A_0$.
For each conditioned jitter/rotation realization the exceptional target set
is null; Fubini preserves the lower bound almost everywhere after mixing.
Integrate the bounded-jitter event and then the actual probabilities of every
pattern. This proves the minorization in (CGD.5), without factoring final
velocity rows.

**3. Bound extinction as well as survival.** On the same jitter event,
restrict the standard OU innovations to row norm at most $G$, with probability
$p_G^N$. Then $|v_{2,i}|\le Z_0$ and $|x_{2,i}|\le B_x+tZ_0$.
Independently for every row, the final position noise has probability at least
$1-\Phi((L_D+B_x+tZ_0)/s)$ of putting its first coordinate above $L_D$.
All rows are then dead. Their product gives $\zeta_N$. Survival is at least
the minorization weight. This proves the survival bounds.

**4. Verify compactness and obtain the spectral data.** For a fixed pattern
and fixed jitters/rotations, {prf:ref}`thm-cgd-phase-smoothing` gives TV
continuity with respect to the effective input. Bounded convergence over
jitters and compact rotations preserves it. There are finitely many patterns,
whose probabilities extend continuously on each status component of $K_N$.
The cap, terminal marking and removal of extinction contract total variation.
Thus $k\mapsto Q_N^\Theta(k,\cdot)$ is TV-continuous, and Arzelà–Ascoli
makes its action on $C(K_N)$ compact. Full support from
{prf:ref}`thm-cgd-phase-smoothing` makes it strongly positive. Its spectral
radius is at least $\epsilon_N$, by iteration of the survival lower bound.

The hypotheses of [Zhang, Theorem 1.1](https://arxiv.org/pdf/1606.04377)
are now verified: the cone of nonnegative continuous functions is total and
has nonempty interior, the operator is compact and strongly positive, and
its spectral radius is positive. It supplies the positive eigenfunction and
the eigenvalue. Taking the minimum and maximum of the eigenfunction in its
eigen-equation bounds the eigenvalue by the survival bounds in (CGD.5).

**5. Prove uniqueness and conditioned mixing.** Define

$$
P_N^\Theta(k,dl)
=\frac{Q_N^\Theta(k,dl)e_N^\Theta(l)}{
\alpha_N^\Theta e_N^\Theta(k)}.
$$

It is Markov and minorizes
$\delta_N^\Theta\widehat\theta_0$, where
$\widehat\theta_0=e_N^\Theta\theta_0/\theta_0(e_N^\Theta)$.
Subtracting this common part contracts total variation by
$1-\delta_N^\Theta$. Iteration yields its unique invariant law $\pi_N^\Theta$.
The probability proportional to $(e_N^\Theta)^{-1}\pi_N^\Theta$ is the QSD.
Any other QSD has the same eigenvalue by integration against $e_N^\Theta$,
and its normalized $e_N^\Theta$-weighted law is invariant for $P_N^\Theta$;
therefore it is equal to this QSD. The identity

$$
\eta(Q_N^\Theta)^nf=(\alpha_N^\Theta)^n
\eta\left[e_N^\Theta(P_N^\Theta)^n(f/e_N^\Theta)\right]
$$

and the bounds $m_N^\Theta\le e_N^\Theta\le1$ give (CGD.6) exactly as in
the canonical proof.

**6. Recover every retained physical coordinate.** The effective-coordinate
map is a measurable bijection on actual finite-coordinate states, with the
same inverse dead-position feature as the canonical proof. The two processes
intertwine under identical innovations. All output coordinates are finite;
the QSD eigenmeasure identity consequently puts no mass on artificial
compactification points. Its inverse reconstructs the complete physical QSD
and preserves the TV estimates. $\square$
:::

:::{prf:corollary} Verified QSD of the unchanged viscous reference gas
:label: cor-cgd-reference-qsd

The complete existing parameter instance
{prf:ref}`def-cgd-existing-reference` has a unique full marked QSD and the
conditioned convergence estimate (CGD.6). This conclusion requires no
additional landscape hypothesis. In its actual parameters,

$$
t=0.02,\quad c=e^{-0.04},\quad
q^2=(1-e^{-0.08})/2>0,\quad s^2=0.0004>0,
\quad V_c=4,\quad\ell_K=e^{-1/2},
$$

$$
\alpha=0.9996,\qquad t\nu=0.006,\qquad
\kappa_{\rm count}=0.9936>0.
$$

The declared row-normalized configuration with these same existing parameter
values also satisfies $\kappa_{\rm row}=0.9876>0$ and hence has the same
qualitative finite-$N$ conclusions, with its own explicit certificate.
For an immediately evaluable certificate in either case set
$J=0.1$, $G=1$, $L_0=1$ and $r_v=1$ in (CGD.4). Then

$$
p_J=p_G=G_3(1),\quad B_0=2\sqrt3+0.1,\quad
B_1=4.048+0.02B_0,\quad B_x=B_0+0.02B_1,\quad Y=2,
$$

and $Z,D_T,g_*,J_*,\epsilon_N,\zeta_N$ are the explicit expressions
(CGD.4) with $d=3,N=200$ and the displayed constants. These bounds may be
very small; their role here is the proved finite-population certificate.
:::

:::{prf:proof}
All canonical donor and fitness regularizers are strictly positive in the
displayed configured tuple. Its quadratic force has $\lambda=1$. Substitution
gives the two calculated positive margins and the positive two noise
amplitudes. Thus every condition of {prf:ref}`thm-cgd-finite-n-qsd` is verified
from the existing configuration. The four analysis radii lie strictly within
the required intervals, and $J/\sigma_J=G=1$. They select positive-probability
events in the proof and leave the actual Gaussian innovations unchanged.
$\square$
:::

### 5.1. Analytic force profiles and closed eigenfunction bounds

:::{div} feynman-prose
Why look at two updates? In one update, changing the OU draw also changes
the position where the final force is evaluated. That map can fold, and its
density can become large near a critical point. The first update leaves a
fresh Gaussian position draw for the second update. For each possible donor
pattern, retained positions inherit that noise; copied positions have their
own clone jitter. We can then hold the second-stage positions fixed while
changing velocity. The velocity map is affine in those coordinates. Under
the stated derivative margins, this gives an upper density bound on the
favorable part of the second update, without assuming away the one-update
critical points.

Now normalize the survival eigenfunction to have maximum one. From a state
at that maximum, two updates must carry a definite amount of eigenfunction
mass. The upper density bound prevents all that mass from hiding in an
arbitrarily tiny region. The one-update lower density then lets every starting
state reach the same region, forcing the eigenfunction's minimum above zero.
That is how the unknown ratio becomes a bound calculated from parameters.

The favorable event is a device for estimating probability. Every Gaussian
draw remains unbounded; the exceptional draws remain in the kernel and are
accounted for by a tail-probability bound. Nor do we require fitness values to differ:
donor-pattern probabilities are bounded above by one, never divided out.
The force and noise margins, rather than fitness separation, determine where
this certificate applies.
:::

:::{prf:definition} Actual force profiles for the quantitative extension
:label: def-cgd-analytic-force-profile

Retain the entire canonical marked kernel, terminal box, cap, two Gaussian
viscous normalizations and passive-record convention of
{prf:ref}`def-cgd-parameter-register`. For its configured force define

$$
B_F=|F(0)|,\qquad L_F=\sup_{x\in\mathbb R^d}\|DF(x)\|_{2\to2}.
$$

The extension below concerns globally defined real-analytic $F$ with finite
$L_F$. These profiles must be calculated from the original force; a divergent
profile is not replaced by an assumed finite constant. Set $\nu_N=\nu$ for
$N\ge2$ and $\nu_N=0$ for $N=1$, and define

$$
\kappa_F=1-t^2L_F-\chi t\nu_N>0,\qquad
\chi=\begin{cases}1,&\mathfrak n=\mathrm{count},\\
2,&\mathfrak n=\mathrm{row}.
\end{cases}
\tag{CGD.8}
$$

For an analysis jitter radius $J_0>0$, put $p_0=G_d(J_0/\sigma_J)$ if
$\sigma_J>0$ and $p_0=1$ otherwise, and

$$
\begin{aligned}
B_0(J)&=R_D+J,\\
B_1(J)&=(1+2t\nu_N)V_c+t[B_F+L_FB_0(J)],\\
B_x(J)&=B_0(J)+tB_1(J).
\end{aligned}
$$

For any position cube radius $R>0$ and pre-cap row velocity radius $H>0$, let

$$
Z(H)=\begin{cases}
\sqrt N\{H+t[B_F+L_FB_x(J_0)]\}/\kappa_F,&\mathrm{count},\\
\{H+t[B_F+L_FB_x(J_0)]\}/\kappa_F,&\mathrm{row},
\end{cases}
\qquad R_2(H)=B_x(J_0)+tZ(H).
$$

Define $D_F(H)=1+t^2L_F$ for $N=1$. For $N\ge2$ define

$$
D_F(H)=1+t^2L_F+2t\nu_N+
\begin{cases}
4t^2\nu_N\ell_KZ(H),&\mathrm{count},\\
8t^2\nu_N\ell_KZ(H)e^{2R_2(H)^2/\rho^2},&\mathrm{row},\ N>2,\\
0,&\mathrm{row},\ N=2.
\end{cases}
$$

With $m=dN$, the following is a lower bound for the density in **raw position
and pre-cap velocity coordinates**, before removing extinction:

$$
\mathfrak l_N(R,H)=
\frac{p_0^N(2\pi q s)^{-m}}{(\sqrt N D_F(H))^m}
\exp\left[-\frac{N[Z(H)+cB_1(J_0)]^2}{2q^2}
-\frac{N[\sqrt dR+B_x(J_0)+tZ(H)]^2}{2s^2}\right].
\tag{CGD.9}
$$

Choose $L_0=L_D/2$, $r_v=V/2$ and $H_0=Vr_v/(V-r_v)=V$. A corresponding
one-update common part is

$$
\epsilon_F=\min\left\{\tfrac12,
\mathfrak l_N(L_0,H_0)(2L_0)^m[v_d(r_v)]^N\right\}.
\tag{CGD.10}
$$

The cap's inverse determinant is at least one, explaining the use of capped
target volume in (CGD.10); no such determinant occurs in (CGD.9).
:::

:::{prf:theorem} Finite-population QSD for the actual analytic force
:label: thm-cgd-analytic-force-qsd

Under {prf:ref}`def-cgd-analytic-force-profile`, retain $q,s>0$ and the
canonical continuity and positive regularizers of
{prf:ref}`thm-cgd-finite-n-qsd`. The same full killed marked kernel has a
unique QSD, a continuous eigenfunction $e$ with $\max e=1$, and the
conditioned TV and entropy estimates (CGD.6)--(CGD.7), with its own spectral
data and $\epsilon_F$ in place of $\epsilon_N$.

For either normalization, the primitive survival floor is

$$
\begin{aligned}
\sigma_h&=(t^2q^2+s^2)^{1/2},\\
A&=L_D+J_0+t(1+c)B_1(J_0),\\
a_F&=p_0\left[
\Phi\left(\frac{L_D-A}{\sigma_h}\right)
-\Phi\left(\frac{-L_D-A}{\sigma_h}\right)\right]^d>0.
\end{aligned}
\tag{CGD.11}
$$

Thus $Q1(k)\ge a_F$ and its eigenvalue $\alpha_Q\ge a_F$ uniformly over
all nonextinct capped inputs. Constants need not be uniform in $N$.
:::

:::{prf:proof}
**1. Keep the original B2 force.** For fixed $x_1$, its actual map is

$$
T_{x_1}(z)=z+tF(x_1+tz)+tF^{\mathrm{visc}}(x_1+tz,z).
$$

The global derivative profile gives $|F(x)|\le B_F+L_F|x|$.
The count Laplacian has Euclidean operator norm at most one; the row
Laplacian has maximum-row operator norm at most two. Consequently, in these
respective norms,

$$
\|T_{x_1}(z)\|\ge\kappa_F\|z\|
-t\{\|F(0)\mathbf1\|+L_F\|x_1\|\}.
$$

The same bound holds along the homotopy from this map to $z\mapsto z$.
It is proper, and the degree argument in
{prf:ref}`thm-cgd-phase-smoothing` proves surjectivity. At a consensus $z$,
weight-derivative terms vanish, and

$$
DT_{x_1}(z)=I+t^2\operatorname{diag}(DF(x_{1,i}+tz_0))
-t\nu_NL_{x_1+tz_0}.
$$

The derivative profile and (CGD.8) give an invertible matrix by the Neumann
series in the indicated norm. Its analytic determinant is not identically
zero. The null-critical-set and TV-continuity proof in
{prf:ref}`thm-cgd-phase-smoothing` therefore applies with the original $F$.

**2. Calculate the common density.** On the all-latent-jitter event of radius
$J_0$, all preparation positions and velocities have the bounds $B_0(J_0)$
and $V_c$. B1 and A1 give $B_1(J_0),B_x(J_0)$. The preceding coercivity
bound gives a preimage with each $|z_i|\le Z(H)$. Differentiate the map on
that preimage set. The force derivative contributes $1+t^2L_F$; the viscous
terms are bounded exactly as in (CGD.4), giving $D_F(H)$. For a row-normalized
pair the only nonself weight is one, so its weight derivative is zero.
The maximum-row derivative bound gives determinant at most
$(\sqrt N D_F(H))^m$. Gaussian change of variables now gives (CGD.9), for
every $R,H$ and almost every target in the corresponding raw cube/ball.
Removal of all-dead targets leaves the same lower density on the surviving
targets. The cap then gives (CGD.10).

**3. Compute survival without restricting other rows.** Choose any tagged
row after revival. Its selected original position belongs to the existing
box. On that row's latent-jitter event of radius $J_0$, every coordinate is
at most $L_D+J_0$ in magnitude. Both normalizations give
$|F_i^{\mathrm{visc}}|\le2\nu_NV_c$ at B1 regardless of other jitters, so
$|v_{1,i}|\le B_1(J_0)$. The exact final-position identity is
$x_i^+=x_i+t(1+c)v_{1,i}+tq\xi_i+s\zeta_i$.
Conditional on preparation, its coordinates are independent Gaussians with
standard deviation $\sigma_h$ and means of magnitude at most $A$.
The Gaussian probability of $[-L_D,L_D]$ is minimized at mean $\pm A$:
differentiate its interval integral with respect to the mean to verify this.
Multiply the $d$ coordinate probabilities and the tagged jitter probability.
That row being alive implies survival, yielding (CGD.11), without asserting
independence between final velocities or different preparation rows.

**4. Verify the full eigenproblem.** The effective input compactification,
continuous pattern probabilities, TV-continuity, compactness and strong
positivity checks in {prf:ref}`thm-cgd-finite-n-qsd` are unchanged. Force
evaluation still occurs after mandatory revival and uses no raw dead input
coordinate. The common part is now (CGD.10); a uniform positive extinction
event follows from bounded latent jitter and OU draws followed by sufficiently
large final position draws, as in its Step 3. Its spectral, Doob, uniqueness,
physical-coordinate and entropy arguments apply verbatim. The minimum
eigenfunction equation also gives $\alpha_Q\ge\inf Q1\ge a_F$.
No alteration of force, gate, donor law or noise was used. $\square$
:::

:::{prf:lemma} Two-update upper density with an explicit discarded mass
:label: lem-cgd-two-update-density

Under {prf:ref}`thm-cgd-analytic-force-qsd`, suppose $\sigma_J>0$ when
$N>1$. Choose analysis radii $J,G>0$. Write $\nu_N$ as above and let

$$
\begin{aligned}
\beta_F&=t^2(L_F+C_x)<1,\\
C_x&=\begin{cases}
4\nu_NV_c\ell_K,&\mathrm{count},\\
16\nu_NV_c B_0(J)/\rho^2,&\mathrm{row},\ N>2,\\
0,&\mathrm{row},\ N\le2,
\end{cases}\\
k_B&=1-\chi t\nu_N>0,\qquad
\tau=\begin{cases}\min(s,\sigma_J),&N>1,\\s,&N=1,\end{cases}\\
M_N&=(4N^2)^N(2\pi\tau q)^{-m}
[(1-\beta_F)k_B]^{-m}.
\end{aligned}
\tag{CGD.12}
$$

Set

$$
\begin{aligned}
Z_0&=cB_1(J)+qG,&R'_2&=B_x(J)+tZ_0,\\
H&=(1+2t\nu_N)Z_0+t(B_F+L_FR'_2),&R&=R'_2+sG,\\
p_{\mathrm{bad}}&=N\{\mathbf1_{\sigma_J>0}[1-G_d(J/\sigma_J)]
+2[1-G_d(G)]\}.
\end{aligned}
\tag{CGD.13}
$$

Let $E_N(x,y)=e(x,C_V(y),\mathbf1_D(x))$ on nonextinct physical targets,
and set it to zero on all-dead targets. Let
$\mathcal B=[-R,R]^{dN}\times B(0,H)^N$ in raw position/pre-cap coordinates.
Then, for every effective input $k$,

$$
Q^2e(k)\le M_N\int_{\mathcal B}E_N(x,y)\,dx\,dy+p_{\mathrm{bad}}.
\tag{CGD.14}
$$

The first-update density need not have a bounded Jacobian inverse. All its
critical-point behavior is retained.
:::

:::{prf:proof}
**1. Use the noise at the correct stage.** Condition on all first-update
randomness except its final position innovation. Its complete velocities are
now fixed and capped. The input positions for the second update are independent
Gaussians of standard deviation $s$, with arbitrary centers. Partition the
second-update integral into its incoming masks, measurement donor arrays,
cloning donor arrays and gate arrays: their number is at most
$2^N N^N N^N2^N=(4N^2)^N$. Pattern probabilities depend on the positions,
but each is at most one. Drop those probabilities when bounding an unnormalized
submeasure; do not condition and divide by a pattern probability.

For each pattern, integrate its normalized independent Haar rotations. Given
a rotation realization and the fixed input velocities, the collision velocities
are fixed, independent of continuous position coordinates, and bounded by
$V_c$. A row which does not copy retains its own input position, with density
at most $(2\pi s^2)^{-d/2}$. A copying row, including a revived row, has its
own independent Gaussian jitter, with density at most
$(2\pi\sigma_J^2)^{-d/2}$. Retained rows have distinct original labels.
Condition first on the retained original positions, bound each copying-row
density by its supremum, and integrate unused original positions. This proves
a joint preparation-position density bound $(2\pi\tau^2)^{-m/2}$ for
each pattern, even when donors copy or several recipients share a donor.
For $N=1$, the singleton cannot copy and the first position noise suffices.
Restricting this submeasure to any jitter event cannot increase its density.

**2. Check the first drift, including row normalization.** On the second
update's all-latent-jitter event of radius $J$, every prepared position belongs
to $B(0,B_0(J))$. With fixed collision velocities $v$, the first drift is

$$
A_v(x)=x+tv+t^2[F(x)+F^{\mathrm{visc}}(x,v)].
$$

The count bound for its force's position derivative is
$4\nu_NV_c\ell_K$. For a row-normalized weight and position perturbation
of maximum-row norm $\delta$ on this ball,
$|d\log K_{ij}|\le4B_0(J)\delta/\rho^2$. Differentiating normalized weights
gives $\sum_j|d\omega_{ij}|\le8B_0(J)\delta/\rho^2$. Since
$|v_j-v_i|\le2V_c$, their force derivative is at most
$16\nu_NV_cB_0(J)\delta/\rho^2$. This bound uses the actual ratio and needs
no population-independent Gaussian degree floor. For a pair, that ratio is
identically one; for a singleton viscosity is zero.

Thus $A_v-I$ is $\beta_F$-Lipschitz in maximum-row norm on a convex product
of balls. It is injective there. Every eigenvalue of its Jacobian perturbation
has modulus at most $\beta_F$, so
$|\det DA_v|\ge(1-\beta_F)^m$. Change variables from prepared $x$ to $x_1$,
then use the fresh independent OU density for $z$. The transformation
$(x_1,z)\mapsto(x_2=x_1+tz,z)$ has determinant one. Their joint density is
bounded by
$(2\pi\tau q)^{-m}(1-\beta_F)^{-m}$ per pattern.

**3. Hold the actual second-stage positions fixed.** In coordinates $(x_2,z)$,
B2 is the affine velocity map

$$
y=(I-t\nu_NL_{x_2})z+tF(x_2).
$$

Its count eigenvalues are at least $1-t\nu_N$. Its row Laplacian is similar
to a symmetric matrix with spectrum in $[0,2]$, giving eigenvalues at least
$1-2t\nu_N$. Hence its velocity determinant is at least $k_B^m$.
This argument differs from differentiating the one-update map while keeping
$x_1$ fixed; it does not discard the critical points of that map.
Finally, convolution in $x_2$ with the second final-position Gaussian has
integral one. Sum the unnormalized pattern bounds to obtain density at most
$M_N$ in $(x^+,y)$ on the good-jitter submeasure. Terminal killing only
removes mass.

**4. Retain every exceptional event in the bound.** If all second-update
latent jitters have norm at most $J$ and both standard kinetic innovations
have row norm at most $G$, the maximum-row force bound at both B stages gives
the stage budgets in (CGD.13). The output belongs to $\mathcal B$.
The union bound gives exceptional mass at most $p_{\mathrm{bad}}$; it remains
in the actual kernel. On this part use $0\le e\le1$. On the good part use
the density bound and integrate $E_N$ over $\mathcal B$. This proves
(CGD.14), also at compactification inputs by the established TV-continuity.
$\square$
:::

:::{prf:theorem} Primitive-parameter eigenfunction ratio and conditioned rates
:label: thm-cgd-primitive-eigenfunction

Under {prf:ref}`lem-cgd-two-update-density`, choose its radii so that
$p_{\mathrm{bad}}\le a_F^2/2$. Define

$$
\underline m_F=\min\left\{1,
\frac{\mathfrak l_N(R,H)a_F^2}{2M_N}\right\}>0,
\qquad \underline\delta_F=\epsilon_F\underline m_F>0.
\tag{CGD.15}
$$

The actual full-kernel eigenfunction and Doob common part satisfy

$$
\min e\ge\underline m_F,\quad
\frac{\max e}{\min e}\le\underline m_F^{-1},\quad
\theta_0(e)\ge\underline m_F,\quad
\delta_N\ge\underline\delta_F.
$$

For every initial law $\eta$, its conditioned law obeys

$$
\left\|\frac{\eta Q^n}{\eta Q^n1}-\nu_Q\right\|_{\mathrm{TV}}
\le4\underline m_F^{-2}(1-\underline\delta_F)^n.
\tag{CGD.16}
$$

For every $\eta$ with finite initial relative entropy,

$$
D\left(\frac{\eta Q^n}{\eta Q^n1}\middle\Vert\nu_Q\right)
\le\underline m_F^{-2}(1-\underline\delta_F)^nD(\eta\Vert\nu_Q).
\tag{CGD.17}
$$

Every displayed constant now depends only on the original primitive parameters,
the calculated force profiles and explicit analysis radii. No eigenfunction
minimum, weighted spectral mass or eigenvalue remains an unevaluated input.
No positive cloning-acceptance or fitness-separation premise is used.
:::

:::{prf:proof}
At an effective input attaining $\max e=1$, the eigenfunction equation twice
gives $Q^2e=\alpha_Q^2\ge a_F^2$. By (CGD.14),

$$
\int_{\mathcal B}E_N(x,y)\,dx\,dy\ge\frac{a_F^2}{2M_N}.
$$

For every starting input, the one-update density lower bound (CGD.9) on that
same raw target set gives

$$
\alpha_Qe(k)=Qe(k)\ge\mathfrak l_N(R,H)
\int_{\mathcal B}E_N(x,y)\,dx\,dy
\ge\frac{\mathfrak l_N(R,H)a_F^2}{2M_N}.
$$

Since $\alpha_Q\le1$, (CGD.15) follows. Also
$\delta_N=\epsilon_F\theta_0(e)/\alpha_Q\ge\epsilon_F\underline m_F$.
Substitute these lower bounds into the already proved full-kernel TV and
entropy inequalities. The lower/upper comparison used raw pre-cap coordinates
on both sides and the same $E_N$, so no cap Jacobian or physical coordinate
has been silently dropped. $\square$
:::

:::{prf:corollary} Explicit radii, reference closure, and nonquadratic cases
:label: cor-cgd-primitive-reference

For $\sigma_J>0$ set $J_0=\sigma_J$, compute $a_F$ from (CGD.11), and set

$$
r_*=[2d\{\log(12dN)-2\log a_F\}]^{1/2},\qquad
J=\sigma_Jr_*,\qquad G=r_*.
\tag{CGD.18}
$$

For $\sigma_J=0,N=1$, choose any $J_0>0$, set $J=J_0$ and $G=r_*$.
These choices give $p_{\mathrm{bad}}\le a_F^2/2$. If their derived
$\beta_F<1$ and (CGD.8) holds, all conclusions (CGD.15)--(CGD.17) follow.
The unchanged reference instance satisfies these inequalities for both its
existing count and row normalization choices.

For a configured anisotropic/affine quadratic force $F(x)=-Ax+b$, $A=A^{\mathsf T}$,
$L_F=\|A\|_{2\to2}$ and $B_F=|b|$ give the same certificate. For a configured
potential

$$
U(x)=\tfrac12x^{\mathsf T}Ax-b\cdot x+
\sum_{\ell=1}^d a_\ell[1-\cos(k_\ell x_\ell)],
\quad A=A^{\mathsf T},
$$

the profile bounds are $B_F=|b|$ and
$L_F\le\|A\|_{2\to2}+\max_\ell|a_\ell|k_\ell^2$.
For $\lambda\ge0$ and the nonconvex potential
$U(x)=\lambda|x|^2/2-A_0\sum_\ell\log\cosh(kx_\ell)$,
the force is $-\lambda x+A_0k\tanh(kx)$, with $B_F=0$ and
$L_F\le\lambda+|A_0|k^2$. Each example concerns that force **if it is the
configured force**; it does not authorize changing a run. Their finite-QSD
and primitive rates are discharged precisely where their calculated profiles
pass the displayed inequalities. Global convexity and uniqueness of a landscape
minimum are not additional premises.
:::

:::{prf:proof}
For a standard $d$-Gaussian, $|\xi|>r$ implies at least one coordinate exceeds
$r/\sqrt d$ in absolute value. The elementary exponential Gaussian tail bound
and a union bound give
$1-G_d(r)\le2d\exp[-r^2/(2d)]$. Each of the three terms in (CGD.13) is
therefore at most $a_F^2/6$ at (CGD.18); in the singleton zero-jitter case
only two terms are present. This proves the discarded-mass inequality.

For the reference, $L_F=1,B_F=0,J_0=0.1$ and
$\sigma_h^2=0.02^2q^2+0.02^2$. Substitution in (CGD.11) and (CGD.18)
gives the explicit $J,G$. To verify the row inequality without depending on
a rounded Gaussian tail, note that $B_1(J_0)<4.12$, $A<2.265$ and
$0.02\le\sigma_h<0.021$. Integrating the standard Gaussian density on the
subinterval $[L_D-\sigma_h,L_D]$ gives coordinate survival probability at least
$(2\pi)^{-1/2}\exp[-14.25^2/2]>e^{-103}$. Also $G_3(1)>0.1$, by integrating
the minimum Gaussian density over the unit ball. Hence $-\log a_F<312$,
$r_*<62$ and $J<6.2$. The count derivative margin is
$1-0.02^2[1+4(0.3)(4)e^{-1/2}]>0$.
For row normalization it is
$1-0.02^2[1+16(0.3)(4)(2\sqrt3+J)]>0.925$ by these bounds.
Both coercivity margins were already verified above. Logarithmic diagnostic
evaluation gives $\log a_F\simeq-259.0643$, $J\simeq5.6232$,
$G\simeq56.2322$, and the two first-drift margins approximately $0.99844$
and $0.92981$. These decimal evaluations are supplementary; the inequalities
above prove closure independently of floating-point rounding.
The remaining examples follow by differentiating their original displayed
forces and evaluating the operator norms; $|\cos|\le1$ and
$0<\operatorname{sech}^2\le1$ give the stated global profiles.
$\square$
:::

:::{prf:remark} Quantitative scope and numerical evaluation
:label: rem-cgd-primitive-limitations

The evaluator `fragile.fractalai.theory.qsd_certificate.viscous_qsd_primitive_certificate`
implements (CGD.8)--(CGD.18) in natural logarithms. Positive quantities can be
far below floating-point range; retain their logarithms rather than rounding
them to zero. Its floating calculations are not rigorous interval enclosures
and do not verify the analytic force profile or the canonical kernel contract.

The new conclusion is a finite-$N$ certificate for the existing real-coordinate,
fixed-step, capped, terminal-box killed gas. It does not prove population-uniform
mixing or joint LSI, stationary chaos, an unbounded conservative invariant law,
history-donor or geometry-feedback dynamics, or numerical-kernel transfer.
For $N>1$ the two-update upper-density argument requires existing positive clone
jitter; the earlier quadratic qualitative QSD theorem does not. A nonsmooth or
superlinear force with infinite $L_F$, a nonpositive B2 margin, or a row
first-drift margin failing at the necessary tail radius lies outside this
certificate. Such a failure does not prove nonconvergence of that gas.
The derivative and noise inequalities are sufficient, not necessary.

The constants can be extremely pessimistic: the pattern count, simultaneous
Gaussian events, row derivative bound and minimum-density target comparison
accumulate losses with $N$, small noise or narrow bandwidth. Equal or nearly
equal fitness is not excluded here; the proof bounds pattern probabilities
above by one and never divides by an acceptance probability. The finite Fock
spectral degeneracy question remains separate.
:::

:::{prf:theorem} Population-independent marginal QSD tails and survival
:label: thm-cgd-uniform-marginal-qsd-tails

For the same unchanged quadratic coupled terminal-box kernel of
{prf:ref}`thm-cgd-finite-n-qsd`, keep every primitive parameter in $\Theta$
other than $N$ fixed while varying $N$. The already verified margin
$\kappa>0$ implies $t\nu<1$ for either normalization. Define

$$
\begin{aligned}
a_x&=1-bt\lambda,& A_x&=|a_x|,& R_D&=\sqrt d L_D,\\
M_x&=A_xR_D+bV_c,& s_h^2&=t^2q^2+s^2,&
\tau_x^2&=A_x^2\sigma_J^2+s_h^2.
\end{aligned}
\tag{CGD.8}
$$

For any analysis choice $0<\delta<1/(4\tau_x^2)$ put

$$
B_\delta=\exp(2\delta M_x^2)(1-4\delta\tau_x^2)^{-d/2}.
\tag{CGD.9}
$$

Choose $J>0$ and $0<r_0<L_D$, set
$p_J=G_d(J/\sigma_J)$ for $\sigma_J>0$ and $p_J=1$ otherwise, and define

$$
\eta=p_Jv_d(r_0)(2\pi s_h^2)^{-d/2}
\exp\!\left[-\frac{[r_0+A_x(R_D+J)+bV_c]^2}{2s_h^2}\right]>0.
\tag{CGD.10}
$$

These constants do not depend on $N$, on the accepted graph or on the
retained dead coordinates. For every nonextinct entering state, every row
$i$, and its actual complete update,

$$
\mathbb E[e^{\delta|x_i^+|^2}\mid S]\le B_\delta,
\qquad \mathbb P\{a_i^+=1\mid S\}\ge\eta,
\qquad Q_N^\Theta1(S)\ge\eta.
\tag{CGD.11}
$$

Consequently the actual QSD and its survival eigenvalue satisfy

$$
\begin{aligned}
\alpha_N^\Theta&\ge\eta,\\
\nu_N^\Theta(e^{\delta|x_i|^2})&\le B_\delta/\eta,\\
\nu_N^\Theta\{|x_i|>R\}&\le(B_\delta/\eta)e^{-\delta R^2}
\qquad(R\ge0),\\
\nu_N^\Theta(|x_i|^p)&\le
\frac{B_\delta}{\eta}\Gamma(1+p/2)\delta^{-p/2}
\qquad(p>0),\\
\nu_N^\Theta\!\left(\frac1N\sum_i a_i\right)
&\ge\frac{\eta}{\alpha_N^\Theta}\ge\eta.
\end{aligned}
\tag{CGD.12}
$$

All velocity moments remain bounded by $\nu_N^\Theta(|v_i|^p)\le V^p$.
The tail bounds concern every retained physical row, including dead rows.
They prove uniform marginal tightness. They do not assert a lower bound on
the alive fraction in every swarm, stationary chaos, or a population-independent
mixing rate or joint LSI.
:::

:::{prf:proof}
**1. Isolate a Gaussian marginal without assuming independence of the
bounded viscous term.** Conditional on the frozen companion/gate pattern and
shared rotations, write the post-copy, pre-jitter position of row $i$ as
$x_i^{\rm src}\in\overline D$, its acceptance indicator as $I_i\in\{0,1\}$,
and the post-collision velocity population as $v^{\rm col}$ with
$\max_j|v_j^{\rm col}|\le V_c$. The pattern and rotations precede and are
independent of all clone and kinetic innovations. Thus

$$
x_i=x_i^{\rm src}+I_i\sigma_JZ_{J,i},\qquad
v_1=W_xv^{\rm col}-t\lambda x,
$$

where $W_x$ is the count or row alignment kick matrix. The positive margin
implies $t\nu<1$, so both matrices are row stochastic by
{prf:ref}`lem-cgd-count-kick` and {prf:ref}`lem-cgd-row-kick`. Pathwise,
$|(W_xv^{\rm col})_i|\le V_c$, including all realized jitters. The exact
position formula gives

$$
x_i^+=a_xx_i^{\rm src}+b(W_xv^{\rm col})_i+G_i,\qquad
G_i=a_xI_i\sigma_JZ_{J,i}+tq\xi_{v,i}+s\xi_{x,i}.
$$

Conditional on this frozen pattern and rotations, $G_i$ is a centered
Gaussian of covariance
$(A_x^2I_i\sigma_J^2+s_h^2)I_d\preceq\tau_x^2I_d$. The term
$W_xv^{\rm col}$ can be correlated with $G_i$; the estimate only uses its
pointwise bound. Hence
$|x_i^+|^2\le2M_x^2+2|G_i|^2$. The Gaussian integral
$\mathbb E e^{2\delta|G_i|^2}
\le(1-4\delta\tau_x^2)^{-d/2}$ proves the first inequality in (CGD.11).
It is uniform over every pattern and rotation, so mixing them preserves it.

**2. Use only the tagged jitter to bound survival.** Conditional on all clone
jitters, $x_i^+$ has Gaussian covariance $s_h^2I_d$. On the event
$|\sigma_JZ_{J,i}|\le J$, its conditional mean is bounded by
$A_x(R_D+J)+bV_c$, independently of all other clone jitters. This tagged event
has probability $p_J$ conditional on the frozen pattern and rotations; it may
be imposed also when $I_i=0$. The ball $B(0,r_0)$ lies inside the actual
terminal box. The Gaussian density on that ball is at least

$$
(2\pi s_h^2)^{-d/2}
\exp\!\left[-\frac{[r_0+A_x(R_D+J)+bV_c]^2}{2s_h^2}\right].
$$

Integrating the ball, then the tagged jitter event, and then all remaining
innovations and patterns proves the tagged survival bound $\eta$.
Survival of row $i$ implies nonextinction of the swarm, so $Q_N^\Theta1\ge\eta$.
Integrate against the QSD to obtain
$\alpha_N^\Theta=\nu_N^\Theta Q_N^\Theta1\ge\eta$.

**3. Transfer the unconditional output envelope to the killed QSD.** First
apply the eigenmeasure identity to bounded truncations of
$e^{\delta|x_i|^2}$. Removing extinct outputs only decreases their expectation,
so monotone convergence and (CGD.11) give

$$
\nu_N^\Theta(e^{\delta|x_i|^2})
=\frac{\nu_N^\Theta Q_N^\Theta(e^{\delta|x_i|^2})}{\alpha_N^\Theta}
\le\frac{B_\delta}{\alpha_N^\Theta}\le\frac{B_\delta}{\eta}.
$$

Exponential Markov inequality proves the tail. Integrating it gives
$p\int_0^\infty R^{p-1}e^{-\delta R^2}dR
=\Gamma(1+p/2)\delta^{-p/2}$ and the moment bound. The cap gives the velocity
claim. On an extinct output the alive fraction is zero, so its killed-kernel
expectation equals its full-update expectation. Each row's survival probability
is at least $\eta$; averaging and using the QSD identity proves the last line
of (CGD.12). These estimates control expectations, and do not imply the
stronger population or gradient claims excluded in the statement. $\square$
:::

:::{prf:corollary} Verified marginal tail certificate for the existing reference
:label: cor-cgd-reference-uniform-tails

For the unchanged reference parameters in
{prf:ref}`def-cgd-existing-reference`, and also for its already declared
row-normalized setting, (CGD.8)--(CGD.12) hold with the same constants for
every $N$ for which the reference components are instantiated. In particular,

$$
\begin{aligned}
b&=0.02(1+e^{-0.04}),&
A_x&=1-0.0004(1+e^{-0.04})>0,\\
M_x&=2\sqrt3 A_x+4b,&
s_h^2&=0.0002(1-e^{-0.08})+0.0004,\\
\tau_x^2&=0.01A_x^2+s_h^2,&
\delta&=1/(8\tau_x^2),\\
B_\delta&=2^{3/2}\exp[M_x^2/(4\tau_x^2)].
\end{aligned}
$$

The analysis choices $J=0.1$, $r_0=1$ give the fully explicit uniform constant

$$
\eta=G_3(1)\frac{4\pi}{3}(2\pi s_h^2)^{-3/2}
\exp\!\left[-\frac{[1+A_x(2\sqrt3+0.1)+4b]^2}{2s_h^2}\right].
$$

For scale checking, $A_x\approx0.9992156842$,
$s_h^2\approx0.0004153767307$, $\tau_x^2\approx0.01039969657$ and
$\delta\approx12.01958146$. The lower survival bound is positive but
conservative; the displayed formulas, rather than rounded values, define the
certificate. No innovation is truncated and no new configured parameter is
introduced.
:::

:::{prf:proof}
The original reference has $t\nu=0.006<1$ and the positive QSD margins
verified in {prf:ref}`cor-cgd-reference-qsd`. Substitute its actual
$\lambda=1$, $L_D=2$, $V_c=4$, $\sigma_J=0.1$ and noise variances into
(CGD.8)--(CGD.10). The chosen $\delta$ lies strictly between zero and
$1/(4\tau_x^2)$, and the ball of radius one is inside the configured box.
Both normalizations give the same convex-kick bound used in this theorem.
Their QSDs and eigenvalues remain distinct laws; only these marginal
certificate constants coincide. $\square$
:::

(sec-coupled-gas-entropy)=
## 6. Full-step entropy and the unresolved uniform LSI

:::{div} feynman-prose
The common part of the transition has a useful information meaning. Whenever
that part is selected, the new state carries no information about which
starting state produced it. Repeating the update therefore erases a fixed
fraction of relative entropy under the conservative Doob law. Reweighting
at the beginning and end translates this estimate back to survival-conditioned
trajectories of the gas.

This gives an entropy theorem for the full coupled update. A full-gradient
logarithmic Sobolev inequality asks a different question: how much entropy can
a function have compared with its spatial and velocity derivatives? Status
marks make the distinction especially concrete. A function can change between
alive/dead patterns while every continuous derivative stays zero. The existing
status decomposition records the additional discrete contribution, and the
finite-population mixing proof does not remove it.
:::

:::{prf:theorem} Entropy convergence of the actual coupled killed gas
:label: thm-cgd-discrete-entropy

Under {prf:ref}`thm-cgd-finite-n-qsd`, for every law $\mu$ with finite
$D(\mu\Vert\nu_N^\Theta)$,

$$
D\!\left(\frac{\mu(Q_N^\Theta)^n}{\mu(Q_N^\Theta)^n1}
\middle\Vert\nu_N^\Theta\right)
\le\frac1{(m_N^\Theta)^2}(1-\delta_N^\Theta)^n
D(\mu\Vert\nu_N^\Theta).
\tag{CGD.7}
$$

All components of the declared full transition, including both coupled
viscous kicks, are inside this inequality. No joint full-gradient LSI is
assumed or inferred.
:::

:::{prf:proof}
Use the actual Doob kernel and invariant law constructed in the preceding
proof. Its minorization gives
$P_N^\Theta=\delta_N^\Theta\widehat\theta_0+
(1-\delta_N^\Theta)R_N$ with $R_N$ Markov. Joint convexity and data
processing imply

$$
D(\eta P_N^\Theta\Vert\pi_N^\Theta)
\le(1-\delta_N^\Theta)D(\eta R_N\Vert\pi_N^\Theta R_N)
\le(1-\delta_N^\Theta)D(\eta\Vert\pi_N^\Theta).
$$

The full conditioned iterate is exactly
$\mathcal R_{1/e_N^\Theta}[(\mathcal R_{e_N^\Theta}\mu)(P_N^\Theta)^n]$.
Each bounded reweighting costs at most $1/m_N^\Theta$ by
{prf:ref}`lem-kl-bounded-reweighting`. Iterate and multiply the two costs,
which proves (CGD.7). This is the full proof of
{prf:ref}`thm-hypocoercive-canonical-discrete-entropy` with its previously
zero-viscosity kernel hypotheses now verified for the coupled map. $\square$
:::

:::{prf:corollary} Verified entropy convergence of the viscous reference gas
:label: cor-cgd-reference-entropy

For the unchanged reference instance in
{prf:ref}`def-cgd-existing-reference`, the full marked survival-conditioned
law satisfies (CGD.7) for every initial law of finite entropy relative to its
actual QSD, with the same explicit minorization certificate and the identified
eigenfunction quantities. No new functional-inequality assumption is required.
:::

:::{prf:proof}
The complete QSD and minorization conditions of
{prf:ref}`thm-cgd-discrete-entropy` are proved for this instance in
{prf:ref}`cor-cgd-reference-qsd`. Substitute them into its complete-kernel
entropy proof. $\square$
:::

:::{prf:proposition} What the finite-population certificate does not supply
:label: prop-cgd-lsi-scope

The constants (CGD.4)--(CGD.7) do not establish an $N$-uniform joint
full-gradient LSI, an $N$-uniform entropy rate, stationary chaos, or an
unbounded-domain QSD. In particular, a full-gradient inequality on the
status-bearing QSD cannot use only continuous derivatives when more than one
status stratum has positive probability. A separate discrete form is needed.

A stratumwise density-comparison route would require actual bounds
$0<a_{N,\mathfrak a}\le d\nu_{N,\mathfrak a}^\Theta/dm_{N,\mathfrak a}
\le b_{N,\mathfrak a}<\infty$ against a specified LSI reference, followed by
the status entropy term of {prf:ref}`prop-kl-status-entropy`. No such
two-sided global bounds are conclusions of the nonlinear smoothing theorem.
:::

:::{prf:proof}
Formula (CGD.4) contains product probabilities and Gaussian densities in
dimension $dN$, and no positive $N$-uniform lower bound is supplied. The
new primitive eigenfunction bound (CGD.15) is finite-$N$ and still has no
positive population-uniform lower bound. A minorization
controls Markov entropy contraction; it supplies no derivative form for the
invariant measure. The nonlinear map in (CGD.2) can have critical points;
a bounded upper Jacobian used for a density lower bound is not a lower bound
on its Jacobian or a global density upper bound.
For the status assertion take a function constant on each continuous stratum
but with different constants on two strata of positive mass. Its continuous
gradient is zero and its entropy is positive. This is exactly the discrete
term in {prf:ref}`prop-kl-status-entropy`. Finally the QSD proof uses actual
revival into a bounded terminal domain, and therefore cannot be applied with
$D=\mathbb R^d$. $\square$
:::

(sec-coupled-gas-readout)=
## 7. Color and geometry readouts of the proved law

:::{div} feynman-prose
At the first kick, a consensus population has zero viscous force. A color
formula that divides by its magnitude must retain the specified zero mask
there. At the second kick, the independent OU innovations have spread the
velocities. The analytic calculation below shows that an exactly zero force
then has probability zero for each row. Thus the recorded B2 color direction
is defined almost surely in the three-dimensional reference gas.

A geometry readout can also accompany this law. Specify how the positions
produce the graph, metric and any fallback values, then record those outputs.
The resulting law is the pushforward of the proved gas law. This statement
accounts for passive recorded geometry; if a metric feeds back into rewards,
forces or noise, its changed transition requires its own estimates. The stage
and eligible pool used by a tessellation remain part of the readout definition.
:::

:::{prf:theorem} Nonzero native viscous color force and measurable geometry lift
:label: thm-cgd-readout-law

Under {prf:ref}`thm-cgd-finite-n-qsd`, the full physical QSD is absolutely
continuous on every nonempty terminal status stratum. If $N\ge2$, $\nu>0$
and the readout computes the Gaussian viscous force from all finite recorded
slots, then for every row

$$
\nu_N^\Theta\{F_i^{\rm visc}(x,v)=0\}=0.
$$

Thus the direction normalization needed for its native three-component color
readout is almost surely defined when $d=3$. For the *actual recorded B2
force*, rather than a terminal recomputation, the stronger stage-specific
statement holds: conditional on every finite A1 input, for each row

$$
\mathbb P\{F_i^{\rm visc}(x_1+tv_2,v_2)=0\mid x_1,v_1\}=0,
\qquad v_2=cv_1+q\xi_v,
$$

when $N\ge2$, $\nu>0$ and $q>0$. It therefore remains zero after conditioning
on survival under the actual QSD. This is the recorded B2 color existence
statement of {prf:ref}`thm-variant-b2-color-nondegeneracy` with its input law
now supplied by {prf:ref}`thm-cgd-finite-n-qsd`. A B1 force can vanish at
consensus and retains its declared zero mask. No conclusion here identifies
non-Abelian curvature or a gauge-sector mass gap.

For any declared measurable $\Gamma$, augment the state by its deterministic
readout. The law $(\mathrm{id},\Gamma)_\#\nu_N^\Theta$ is the unique QSD on
the consistent augmented states, with the same survival eigenvalue and
conditioned TV/entropy bounds. Geometry parameters affect this pushforward,
not the constants of the unchanged dynamics.

If the geometry uses a fixed coordinate projection to $d'$ dimensions and
all $N\ge d'+2$ terminal recorded slots, repeated sites, affine degeneracies
and $(d'+2)$-site cospherical degeneracies have probability zero. A readout
using only alive slots must additionally define its behavior when that pool
has fewer than $d'+1$ sites.
:::

:::{prf:proof}
Every one-step output on a status stratum is absolutely continuous by
{prf:ref}`thm-cgd-phase-smoothing`; mixtures preserve this property.
The eigenmeasure identity $\nu_N^\Theta=(\alpha_N^\Theta)^{-1}
\nu_N^\Theta Q_N^\Theta$ proves it for the QSD.

For a fixed row the force is a real analytic function on all finite
coordinates (the row denominator is strictly positive). It is nontrivial:
at identical positions, set that row's velocity to a nonzero vector and all
other velocities to zero. Its force is nonzero for either normalization.
At least one real component is therefore a nontrivial real analytic function;
its zero set is null by the analytic argument in the smoothing proof. The
vector zero set is a subset of this null set. Absolute continuity proves the
probability assertion. If $d=3$ this supplies the nonzero direction in the
declared color formula, without assigning any dynamics to its orbit space.
For the B2 statement, fix $x_1,v_1$. The whole vector $v_2$ has a strictly
positive Gaussian density. The function
$z\mapsto F_i^{\rm visc}(x_1+tz,z)$ is real analytic. Set $z_i=u\ne0$
and all other rows to zero: its value is a strictly negative scalar multiple
of $u$, because every Gaussian edge weight is positive. Thus one component
is nontrivial and its zero set is null. This repeats the stage argument of
{prf:ref}`thm-variant-b2-color-nondegeneracy`. Conditioning on a
positive-probability survival event preserves a probability-zero event.

The map $z\mapsto(z,\Gamma(z))$ is a measurable bijection onto the consistent
augmented states, with inverse projection. Its kernels intertwine exactly,
so QSD eigenmeasures correspond, survival probabilities agree, and uniqueness,
TV and entropy estimates are preserved by this bijection.

The projected position law is absolutely continuous because a coordinate
projection of an absolutely continuous finite-dimensional law is absolutely
continuous. Coincidence, loss of affine rank and cosphericity are the zeros of
the familiar coordinate-difference, affine-determinant and sphere-determinant
polynomials; each relevant polynomial is nontrivial. There are finitely many
index subsets, and each zero set is null. The terminal kernel gives positive
mass to every feasible nonempty status pattern, so an alive-only readout
cannot infer a minimum pool size from nonextinction. $\square$
:::
