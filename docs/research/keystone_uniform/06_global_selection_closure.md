# Exact nonlinear selection feedback and the global closure question

(sec-kugc-fixed-kernel)=
## Fixed native kernel and a strict-gap pair

This record keeps the original quadratic reference, both native viscosity
normalizations, both kicks, sampled fitness, component Haar collisions,
unbounded recipient jitter and kinetic innovations, cap and retained terminal
dead coordinates. It addresses the proposed globally negative signed paired
quadratic ledger. A transient failure of that ledger does not establish
failure of delayed mixing or of a population-uniform QSD eigenfunction bound.

:::{prf:definition} Strict-gap native two-row family
:label: def-kugc-two-row

Take the canonical Viscous Euclidean Gas in $d=3$, $N=2$, with

$$
h=0.04,\quad t=h/2,\quad c=e^{-h},\quad
b_h=t(1+c),\quad A=1-b_ht,\quad
\nu=0.3,\quad\rho=1,\quad U(x)=|x|^2/2,
$$

and all the original reference preparation parameters. Let

$$
S_a:\quad x_1=-ae_1,\quad x_2=be_1,\quad
v_1=v_2=0,\quad a_1=a_2=1,\qquad b=0.9,
\quad1\le a\le1.001.
$$

Here the notation $a_1,a_2$ denotes alive marks; $a$ is the physical
radius parameter. The boundary box remains $[-2,2]^3$. Since every
measurement row has exactly one distinct eligible companion, both raw
diversity measurements are the same smoothed norm
$\sqrt{|z_1-z_2|^2+10^{-6}}$. Their actual global standardization is
exactly zero, without replacing sampled diversity by an expectation.
The reward difference and standardized values are

$$
D(a)=\frac{a^2-b^2}{2},\qquad
z(a)=\frac{D(a)}{\sqrt{D(a)^2+0.04}},\qquad
Z_{r,1}=-z(a),\quad Z_{r,2}=z(a).
$$

Let $G(z)=2/(1+e^{-z})+0.1$ and $\varepsilon=10^{-6}/1.1$.
The two actual retained fitnesses are

$$
F_1(a)=1.1G(-z(a)),\quad F_2(a)=1.1G(z(a)),
$$

and the sole possible accepted edge is $1\to2$, with probability

$$
p(a)=\frac{G(z(a))-G(-z(a))}{G(-z(a))+\varepsilon}
=\frac{2.2-2f(a)}{f(a)+\varepsilon},\qquad f(a)=G(-z(a)).
\tag{KUGC.1}
$$

This gate is strictly between zero and one on the whole stated interval.
At $a=1$,

$$
F_1=0.9775734716359625,\quad F_2=1.4424265283640378,
\quad p=0.4755167715766428.
$$

The fitness gap is $0.4648530567280753$ and the gate is unsaturated.
This example therefore does not use a complete or near fitness tie.
:::

:::{prf:lemma} Exact full-step position law
:label: lem-kugc-position-law

For either viscosity normalization, the actual output positions are

$$
x_i^+=AX_i+tq\xi_i^O+s\xi_i^x,\qquad
q^2=(1-e^{-2h})/2,\quad s^2=0.01h,
\tag{KUGC.2}
$$

where row two always has $X_2=be_1$, and row one has
$X_1=-ae_1$ with probability $1-p(a)$ or
$X_1=be_1+0.1\zeta_1$ with probability $p(a)$.
All displayed Gaussian vectors are independent within a marginal.
Consequently the row-one first-coordinate law is the exact mixture

$$
\rho_a=(1-p(a))\mathcal N(-Aa,\sigma_0^2)
             +p(a)\mathcal N(Ab,\sigma_1^2),
$$

$$
\sigma_0^2=t^2q^2+s^2=0.0004153767307226729,\qquad
\sigma_1^2=A^2(0.1)^2+\sigma_0^2=0.010399696566721815.
\tag{KUGC.3}
$$

*Proof.* When there is an accepted component, both frozen velocities
are zero. Its component mean and relative velocities are zero, so
its shared Haar collision output is zero for every Haar realization.
Without an accepted edge both velocities also remain zero. Thus the
first viscous force is exactly zero in either normalization. The first
force kick is $-tX_i$, the two drifts and the OU stage give
$(1-t^2(1+c))X_i+tq\xi_i^O$, and final position diffusion adds
$s\xi_i^x$. The second potential and viscous kicks, followed by the
velocity cap, change velocities only. Terminal classification attaches
the mark and retains these positions even on exit. This proves (KUGC.2).
The Gaussian jitter is independent of the gate and of both kinetic
innovations; integrate it exactly to obtain (KUGC.3).
$\square$
:::

(sec-kugc-quadratic-obstruction)=
## Exact signed obstruction to a globally negative one-step quadratic ledger

:::{prf:theorem} The prescribed common-source quadratic ledger cannot contract globally
:label: thm-kugc-source-quadratic-obstruction

Couple $S_1$ and $S_{1+\delta}$ by the actual common-source maximal
plan, shared accepted-row jitter, shared component rotations where
components agree, and shared kinetic Gaussians, for $0<\delta\le0.001$.
Let the normalized positional cost be
$P^+=\tfrac12\sum_{i=1}^2|x_i^+-\widetilde x_i^+|^2$.
The exact full-step expectation is

$$
\mathbb EP^+=\frac{A^2}{2}\left[
(1-p(1+\delta))\delta^2+
(p(1+\delta)-p(1))\big((1+b)^2+3(0.1)^2\big)
\right].
\tag{KUGC.4}
$$

For its first-coordinate projection, replace $3(0.1)^2$ by $(0.1)^2$.
The entering positional cost is $\delta^2/2$. Direct differentiation gives

$$
p'(1)=4.90316448668362,\qquad
\lim_{\delta\downarrow0}\frac{\mathbb EP^+}{\delta}
=8.909766764728248>0.
\tag{KUGC.5}
$$

For every fixed positive definite paired phase quadratic form
$G=\begin{psmallmatrix}\alpha&\beta\\\beta&\gamma_P\end{psmallmatrix}$,
with minimum eigenvalue $\lambda_->0$, and every nonnegative terminal
status cost, its complete-output expectation is at least
$\lambda_-\mathbb EP^+$. Its input value is $\alpha\delta^2/2$.
Hence it cannot have nonpositive drift at every entering pair under
this prescribed coupling. Every signed recipient, incoming donor,
collision, force, cap and terminal term of the exact ledger sums to
this positive output cost; no refinement of an upper-bound constant
can reverse its actual sign.

*Proof.* Monotonicity of $p(a)$ follows from (KUGC.1). Common acceptance
has mass $p(1)$ and places both prepared populations at the same positions
with the same jitter; their full outputs coincide. Common persistence
has mass $1-p(1+\delta)$ and leaves a row-one prepared difference
$\delta e_1$. Its final position difference is $A\delta e_1$.
The residual plan has mass $p(1+\delta)-p(1)$ and pairs cloning
from $be_1$ with persistence at $-e_1$. Its final row-one difference
is $A[(1+b)e_1+0.1\zeta_1]$. Row-two final positions coincide
in every branch. Integrate the unrestricted jitter second moment
and apply (KUGC.2); this gives (KUGC.4).

For $f(z)=0.1+2/(1+e^z)$,

$$
\frac{dp}{dz}=-\frac{(2.2+2\varepsilon)f'(z)}{(f(z)+\varepsilon)^2},
\qquad \frac{dz}{da}=\frac{0.04a}{[D(a)^2+0.04]^{3/2}}.
$$

Substitute $a=1$ to obtain (KUGC.5). Positive definiteness gives
$Q\ge\lambda_-P^+$ pointwise irrespective of the actual output
velocities, so the B2 viscous kick, cap and velocity cross term cannot
remove the lower bound. Terminal status penalties are nonnegative
and initial marks agree. Since the positive linear term dominates
every fixed input quadratic coefficient as $\delta\downarrow0$,
the claimed global drift closure is impossible for this coupling.
$\square$
:::

:::{prf:theorem} Gaussian-first coupling also does not give global one-step Euclidean contraction
:label: thm-kugc-optimal-one-step-obstruction

The actual first-coordinate transition mean is

$$
m(a)=A[-a(1-p(a))+bp(a)],\qquad
m'(1)=A[-(1-p(1))+(1+b)p'(1)]=8.784633961156265.
\tag{KUGC.6}
$$

For every coupling of the two full output laws, even one formed after
their Gaussian mixtures have been integrated,

$$
\frac12\mathbb E\sum_i|x_i^+-\widetilde x_i^+|^2
\ge\frac12|m(1+\delta)-m(1)|^2.
$$

Thus the optimal normalized positional $W_2^2$ ratio to its input
cost has lower limit at least

$$
\liminf_{\delta\downarrow0}
\frac{W_{2,\rm pos}(P_2(S_1,\cdot),P_2(S_{1+\delta},\cdot))^2}
{\delta^2/2}\ge[m'(1)]^2=77.16979383150002>1.
\tag{KUGC.7}
$$

*Proof.* Integrate (KUGC.3), then differentiate the explicit mean.
For any coupling, Jensen applied to the row-one first-coordinate
difference gives the displayed lower bound. The derivative limit
proves (KUGC.7). Adding velocities or marks to an ordinary Euclidean
phase/status transport cost cannot decrease this projection lower
bound. $\square$
:::

(sec-kugc-smoothing-cost)=
## Explicit smoothing-aware transport and a positive concave-cost calculation

:::{prf:lemma} Exact Gaussian-mixture tangent transport cost
:label: lem-kugc-gaussian-tangent

Let $C_a$ and $\rho_a$ be the one-dimensional mixture CDF and density
in (KUGC.3). The squared $W_2$ tangent norm of this density family is
the finite explicit integral

$$
J(a)=\int_{\mathbb R}\frac{[\partial_a C_a(x)]^2}{\rho_a(x)}\,dx,
\quad
\partial_aC_a=p'(a)(C_1-C_0)+(1-p(a))A\varphi_0,
\tag{KUGC.8}
$$

where $\varphi_0$ is the $\mathcal N(-Aa,\sigma_0^2)$ density and
$C_0,C_1$ are the two Gaussian CDFs. Its lower bound is
$J(a)\ge[m'(a)]^2$. With

$$
\mathcal X(a)=\frac{\sigma_1^2}
 {\sigma_0\sqrt{2\sigma_1^2-\sigma_0^2}}
\exp\left[\frac{A^2(a+b)^2}{2\sigma_1^2-\sigma_0^2}\right],
$$

an entirely explicit upper bound is

$$
J(a)\le\frac{2p'(a)^2\sigma_1^2}{p(a)}[\mathcal X(a)-1]
+\frac{2(1-p(a))^2A^2}{p(a)}\mathcal X(a).
\tag{KUGC.9}
$$

At the native pair $a=1$, the logarithm of this upper bound is
$178.8870500829268$, while the lower bound is $77.16979383150002$.
The Gaussian-first coupling restores quadratic local regularity
instead of the source-first linear cost, but its actual one-step
derivative still exceeds one.

*Proof.* Differentiating the density continuity equation in $a$
gives velocity $v_a=-\partial_aC_a/\rho_a$. Equivalently, for its
quantile $T_a(u)=C_a^{-1}(u)$,
$\partial_aT_a(u)=-\partial_aC_a(T_a(u))/\rho_a(T_a(u))$.
Changing variables $u=C_a(x)$ gives (KUGC.8). The Gaussian tails,
positive mixture weights and $\sigma_1>\sigma_0$ imply integrability.
Moreover $m'(a)=\int v_a\rho_a$, so Cauchy--Schwarz gives the lower
bound. Completing the square in the two Gaussian densities gives
$\int\varphi_0^2/\varphi_1=\mathcal X(a)$.

For completeness the scalar Gaussian negative-Sobolev estimate is

$$
\int(C_0-C_1)^2/\varphi_1\le
\sigma_1^2\int(\varphi_0/\varphi_1-1)^2\varphi_1
=\sigma_1^2[\mathcal X(a)-1].
$$

Indeed integration by parts identifies its left side with the
variational supremum of
$2\int\phi(\varphi_0-\varphi_1)-\int|\phi'|^2\varphi_1$.
Cauchy--Schwarz and the Gaussian Poincare inequality give the bound.
The latter follows directly from the Mehler semigroup $T_t$ for
$\mathcal N(m,\sigma_1^2)$: differentiate its variance,
$\operatorname{Var}\phi=2\sigma_1^2\int_0^\infty
\int|(T_t\phi)'|^2\varphi_1\,dt$, use
$(T_t\phi)'=e^{-t}T_t\phi'$, Jensen and invariance, and integrate
$2\int_0^\infty e^{-2t}dt=1$. These calculations first apply to
smooth bounded functions and extend by truncation.
Finally $\rho_a\ge p(a)\varphi_1$ and $(u+v)^2\le2(u^2+v^2)$
give (KUGC.9). $\square$
:::

:::{prf:theorem} A fully computed positive full-marked concave-cost regime
:label: thm-kugc-concave-local

Use the actual same source/Haar/Gaussian coupling as (KUGC.4) and
the full marked distance

$$
d_{\rm mark}(S,T)^2=\frac12\sum_{i=1}^2
[|x_i-y_i|^2+|v_i-w_i|^2+(a_i-\widetilde a_i)^2].
$$

For the reference pair, every $0<\delta\le2.09\,10^{-5}$ obeys

$$
\mathbb E\,d_{\rm mark}(S_1^+,S_{1+\delta}^+)^{1/2}
\le0.76225\,d_{\rm mark}(S_1,S_{1+\delta})^{1/2}
\tag{KUGC.10}
$$

in either viscosity normalization. This retains unbounded Gaussian
jitter and both kinetic innovations, the physical viscosity kernel,
the velocity cap and terminal mark mismatch.

*Proof.* Write $r=1/2$, $H=t(c+A)$,
$\ell=e^{-1/2}/\rho$, and
$\mu_\chi=2\Gamma((d+1)/2)/\Gamma(d/2)$, the exact mean norm of
$N(0,2I_d)$. On common persistence, first alignment vanishes. The
row-one position difference is $A\delta e_1$ and the OU velocity
difference is $ct\delta e_1$. In row normalization $N=2$ has one
neighbor and $K/Z=1$, so its second viscous force is exactly
$\nu(z_{\rm other}-z_i)$. The resulting squared Lipschitz factor
for the physical normalized phase distance is at most

$$
L_{\rm row}^2=A^2+(H+t\nu ct)^2+(t\nu ct)^2
=0.9999776973498014.
$$

For count normalization, use the actual Gaussian gradient maximum
$\ell$ on its physical separation and the common-OU velocity
$z_2-z_1=ct(1+b)e_1+q(\xi_2^O-\xi_1^O)$. Set

$$
B_0=ct[1+\ell A(1+b)],\qquad B_1=\ell Aq,\qquad
\overline B=B_0+B_1\mu_\chi,\quad
\overline{B^2}=B_0^2+2B_0B_1\mu_\chi+2dB_1^2.
$$

The two raw velocity discrepancies are bounded respectively by
$\delta[H+(t\nu/2)B]$ and $\delta(t\nu/2)B$. The cap is
$1$-Lipschitz. Consequently

$$
\mathbb E L_{\rm count}^2\le A^2+H^2+
2H(t\nu/2)\overline B+2(t\nu/2)^2\overline{B^2}
=1.000043382452391.
$$

These are exact Gaussian first/second moments of the displayed
upper profiles, without a bounded-innovation event.
Common acceptance has zero complete-output discrepancy. On common
persistence, the physical $r$-power cost is therefore at most
$(\delta/\sqrt2)^r(\mathbb EL^2)^{r/2}$.
For terminal marks, only row one's position can differ. Its final
Gaussian coordinate differs by the deterministic shift $A\delta$.
The two boundary crossing intervals have total probability at most
$2A\delta/\sqrt{2\pi\sigma_0^2}$. Subadditivity of the
$r/2$ power adds at most $B_{\rm mark}\delta$ to the marked cost,
where

$$
B_{\rm mark}=2^{-r/2}\frac{2A}{\sqrt{2\pi\sigma_0^2}}
=32.89430560995521.
$$

For the residual source branch, every capped output velocity has
norm below $2$. Its normalized squared velocity discrepancy is at
most $16$, its status cost at most one, and its averaged positional
cost is $A^2[(1+b)^2+3(0.1)^2]/2$. Jensen therefore bounds its
$r$-power cost by

$$
C_R^r=\left[17+A^2((1+b)^2+0.03)/2\right]^{r/2},\qquad
C_R^2=18.817146210151844.
$$

On $1\le a\le1.001$, the gate derivative has the proved upper bound

$$
P_*'=\frac{1.1+\varepsilon}{[f(1.001)+\varepsilon]^2}
\frac{0.04(1.001)}{[D(1)^2+0.04]^{3/2}}
=5.157789805744965.
$$

Here $|df/dz|\le1/2$, $D(a)\ge D(1)$, $a\le1.001$,
and $f(a)\ge f(1.001)$ justify each inequality. Thus residual
mass is at most $P_*'\delta$. Summing all three source branches gives

$$
\mathbb E d_{\rm mark}^r\le
q_r(\delta/\sqrt2)^r+C\delta,\qquad
q_r=(1-p(1))(\mathbb EL^2)^{r/2},\quad
C=B_{\rm mark}+P_*'C_R^r=43.63672440240339.
$$

For count normalization $q_r=0.5244889166729909$; for row
normalization $q_r=0.5244803040574045$.
The explicit radius condition
$\delta\le[(1-q_r)/(2C2^{r/2})]^{1/(1-r)}$ yields contraction
factor $(1+q_r)/2$. Its count value is
$2.0991429436504158\,10^{-5}$ and its row value is
$2.099218985095488\,10^{-5}$. Rounding the radius down and the
factor up gives (KUGC.10). As a direct outward-rounded check of
the displayed elementary formulas, 40-digit interval arithmetic gives

$$
0.76172611400244962308886686478095578250014
\le q_r+C2^{r/2}(2.09\,10^{-5})^{1-r}
\le0.76172611400244962308886686478095578250061
$$

for count normalization. This interval is below $0.76225$; the row
factor is smaller because its $\mathbb EL^2$ is smaller.
$\square$
:::

(sec-kugc-delayed-target)=
## Delayed closure remains a separate mathematical target

:::{prf:remark} Scope of the exact computations
:label: rem-kugc-delayed-target

The complete-kernel calculation disproves a globally negative
one-step quadratic remainder under the prescribed source coupling.
The mean calculation also excludes solving this problem by merely
integrating the Gaussian mixture before applying an ordinary Euclidean
one-step contraction argument. Both statements concern strictly
separated fitnesses in the unchanged algorithm.

The concave-cost calculation is a positive alternative analysis of
that same kernel. It does not change the force, gate, donor law,
noise or cap. Its explicit radius concerns the declared two-row
family. It cannot be iterated as a global theorem: the next noisy
paired states need not lie in that family, velocity differences need
not vanish, and donor copies can transport a larger entering donor
error into several recipients. Global incoming flux and component
influence are therefore still required in a delayed coupling bound.
No invariance of a compact all-alive phase follows from these noises.

In particular these calculations do not establish absence of
population-uniform delayed marginal convergence or a uniform full-QSD
eigenfunction ratio. They establish the exact obstruction to the
proposed one-step signed quadratic closure and a quantitative positive
replacement regime. A global delayed closure would additionally need
an actual multi-step signed incoming-donor calculation valid for every
marked input and a smoothing/concave coupling that controls revival,
terminal marks and full components over those blocks. Neither the
finite-particle QSD theorem nor the already discharged averaged
Keystone recipient pressure supplies that multi-step conclusion.
:::
