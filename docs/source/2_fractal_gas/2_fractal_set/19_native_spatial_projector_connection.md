# A native spatial projector connection in the zero final-noise regime

(sec-nspc-register)=
## 1. Complete execution and the existing zero-noise branch

:::{prf:definition} Native spatial connection register
:label: def-nspc-complete-register

Retain every execution, landscape, algorithm, donor, fitness,
standardization, acceptance, cloning, jitter, collision, force,
cap, boundary, random-stream, arithmetic, history, geometry,
mask and calibration field of
{prf:ref}`def-nmg-complete-register`. The positive regime here
uses its existing quadratic dense count or nonself row gas,
real independent Gaussian O innovations, $d=3$, $h>0$,
$t=h/2$, $q>0$, $\rho>0$, $\nu>0$, and the existing
final-position parameter $\sigma_x=0$. This is a specified
included parameter regime, not the unchanged numerical
reference with $\sigma_x=.1$. Its input is the allowed
all-alive consensus $x_i=v_i=0$.

Every living fitness agrees at this input, every accepted
living clone gate is zero, the original sampled companion
record is retained, and each collision component is a
singleton with zero relative velocity. B1, both initial
positions and first velocities are zero. The actual B2
input and terminal positions are exactly

$$
z_i=qG_i,\qquad X_i=tz_i,\qquad Y_i=X_i=tz_i,
\qquad G_i\ \hbox{independent }N(0,I_3).
\tag{NSPC.1}
$$

The second total force, its actual final velocity cap, and
terminal alive mark $\mathbf1_D(Y_i)$ are still executed.
They do not change (NSPC.1). The color instrument uses the
matched recorded B2 viscous force and its uncapped force-input
velocity $z_i$, with the original force threshold $\delta_c$,
alignment and finite-value marks. An additional clone/source
deletion mask retains its own channel and is not silently
removed. The phase coefficient is its configured
$\kappa=m\ell_0/\hbar_{\rm eff}$.

Use the actual fixed phase coefficient, or retain a consumed
calibration $\kappa_N$ in the same joint subsequence. A finite
limit gives the limit stated below; nonconvergent or diverging
calibrations retain their original outcomes. The explicit
fixed-source first derivatives use fixed or preceding-record
calibration. If a consumed current-source calibration varies
under the source shift, its actual derivative is included.

Geometry remains the existing passive full-dimensional open
Euclidean CSR/Delaunay geometry, with every recorded covariance
ridge, clamp, pseudo-inverse, metric, weight, volume, projection,
precision, allocation and error parameter retained. Its native
projectors and their canonical projection-product transport
are those of Chapter NPC. This transport is algebraically
determined by the retained $P_i=c_ic_i^\dagger$ and ambient
comparisons; it is distinct from literal $x+iv$ ray transport.
No new gauge sample or force is introduced.

The literal Rust experiment-8 archive reader selects matched
`B1` viscous force and `B1` force-input velocity. In this
consensus transition both are zero, so that literal channel
is unavailable for every $N$. The positive spatial results
below concern the matched RECORDED B2 instrument specified
above, a pushforward of the existing executed history.
They do not assign its field to the B1 archive consumer.
:::

(sec-nspc-exact-kernel-interpolant)=
## 2. The executed kernel gives a smooth spatial force function

:::{prf:theorem} Exact native force function and compact derivative convergence
:label: thm-nspc-native-force-interpolant

On the original innovation array define

$$
K(z,z')=e^{-t^2|z-z'|^2/(2\rho^2)},\qquad
A_N(z)=\frac1N\sum_jK(z,z_j),\qquad
B_N(z)=\frac1N\sum_jK(z,z_j)z_j.
$$

The actual force at every native node is the evaluation
at $z=z_i$ of the following existing-kernel functions:

$$
F_N^{\rm count}(z)=\nu[B_N(z)-A_N(z)z],
$$
$$
F_N^{\rm row}(z)=\nu\frac{B_N(z)-A_N(z)z}{A_N(z)-N^{-1}}.
\tag{NSPC.2}
$$

The row expression is used wherever its denominator is
positive; at an executed node for $N\ge2$ it is strictly
positive and is exactly the nonself denominator. The
singleton has its original zero viscous force. The row
subtraction retains the literal self-exclusion, rather
than replacing the algorithm by an all-row normalization.
Away from nodes, (NSPC.2) is a derived analytic function
agreeing at every executed node. It is not assigned the
counterfactual force from moving one addressed root while
keeping every other row fixed. That actual frozen-other-row
force is explicitly

$$
F_N^{{\rm count},i}(z)=F_N^{\rm count}(z)
             -\frac\nu N K(z,z_i)(z_i-z),
$$
$$
F_N^{{\rm row},i}(z)=\nu\frac{
B_N(z)-A_N(z)z-N^{-1}K(z,z_i)(z_i-z)}{
A_N(z)-N^{-1}K(z,z_i)}.
\tag{NSPC.16}
$$

On each positive compact-degree event these tagged functions
are uniformly $O(N^{-1})$ close to (NSPC.2) in $C^2$,
for all $i$. Thus their same spatial limit is also derived
for the actual counterfactual root force, with every
nonself term retained.

For every fixed compact $K_z\subset\mathbb R^3$, both
functions converge in $C^2(K_z)$ in probability to

$$
F^{\rm count}(z)=-\nu a_0r_0e^{-t^2|z|^2/(2B)}z,
\qquad F^{\rm row}(z)=-\nu r_0z,
$$
$$
B=\rho^2+t^2q^2,\qquad
a_0=(\rho^2/B)^{3/2},\qquad r_0=\rho^2/B.
\tag{NSPC.3}
$$

A primitive bound on each compact derivative error is
$O_{\Pr}(\sqrt{\log N/N})$. Its constants are evaluated
Gaussian-polynomial kernel bounds in $t,\rho,\nu,K_z$
and, for row normalization, the positive degree lower
bound $a_0e^{-t^2R_z^2/(2B)}$ with
$R_z=\sup_{z\in K_z}|z|$. The original Gaussian rows
are not clipped. Only executable populations in the
complete record belong to this limit family.
:::

:::{prf:proof}
At a native $z_i$, the self term of $B_N-A_Nz_i$ is
identically zero. In row normalization the self kernel
weight is exactly one, giving the displayed subtraction.
These identities prove exact agreement with the executed
forces and their original denominators.

For count normalization write its summand as
$\nu e^{-a^2|u|^2/2}u$, $u=z'-z$, $a=t/\rho$.
For every derivative order $k\le3$ its global bound is

$$
L_k=\nu a^{k-1}
 \sup_{w\in\mathbb R^3}
       \|D^k(e^{-|w|^2/2}w)\|<\infty.
\tag{NSPC.4}
$$

The supremum is finite because these derivatives are
polynomials times the original Gaussian kernel.
For example $L_0=\nu\rho/(t\sqrt e)$.
For $A_N$ the same bounds use derivatives of the scalar
Gaussian. On $|z|\le R_z+1$ the summands of $B_N$ and
their derivatives are also bounded: write
$K(z,z')z'=K(z,z')z+K(z,z')(z'-z)$ and use (NSPC.4).
Thus every derivative through order two is a mean of
independent bounded original-row functions, and its
spatial Lipschitz constant is bounded by the corresponding
third derivative profile, uniformly in the innovation array.

For a scalar summand bounded by $L$, its centered value
has absolute value at most $2L$. Expansion of its moment
generating function, using
$E|U|^k\le(2L)^{k-2}EU^2$ and
$k!\ge2\,3^{k-2}$ for $k\ge2$, gives the bound

$$
\Pr(|N^{-1}\textstyle\sum_jU_j|>\epsilon)
\le2\exp\left[-\frac{N\epsilon^2}
                   {2L^2+(4L/3)\epsilon}\right].
$$

Cover the compact query set by a mesh of spacing a
fixed primitive multiple of $\epsilon_N$, where
$\epsilon_N=A\sqrt{\log N/N}$ and $A$ is larger than
the finitely many derivative-profile constants. This
mesh has $O(\epsilon_N^{-3})$ points. Union bound over
the finite matrix components and derivative orders
then tends to zero when $A$ is sufficiently large.
The third-derivative bound extends the resulting error
from the mesh to the entire compact set. This proves
the stated $C^2$ rate for $A_N,B_N$ and the count force.

Completing the original $q$-Gaussian square gives
$A(z)=a_0e^{-t^2|z|^2/(2B)}$ and
$B(z)/A(z)=(t^2q^2/B)z$. In particular $A$ is bounded
below by the displayed positive compact degree bound.
With probability tending to one $A_N-N^{-1}\ge A/2$
on that compact set. Twice differentiating the literal
quotient in (NSPC.2) therefore passes the same rate to
the row force and gives (NSPC.3). The $N^{-1}$ self
correction is retained and vanishes at the proved rate.
All estimates concern the original innovation array.

For (NSPC.16), delete the exact old addressed row from
the empirical count numerator and, in row normalization,
from its exact kernel mass as well. This is precisely
the force of the actual unchanged other rows against
the moved root. Its deleted numerator and derivatives
are bounded by the global profiles (NSPC.4) divided by
$N$. The row mass difference and its derivatives are
bounded by the scalar Gaussian profiles divided by $N$.
On the same degree event both denominators are bounded
below by $A/2$ for sufficiently large $N$. The twice
differentiated quotient rule gives the uniform
$O(N^{-1})$ comparison. It does not replace the full
simultaneous-source derivative by a frozen-row derivative.
:::

:::{prf:theorem} Native spatial projector field on strict available domains
:label: thm-nspc-spatial-projector-field

Let $K_y$ be a compact spatial set whose neighborhood
is inside the actual open domain $D^\circ$, avoids
$y=0$, and has a strict population force margin
$|F^{\mathfrak n}(y/t)|>\delta_c$. Then the native
projector function obtained from (NSPC.2) is available
on that neighborhood with probability tending to one,
and it agrees with every actual available native node there.
For fixed $\kappa$, or $\kappa_N\to\kappa$ in probability,
it converges in $C^2(K_y)$ in probability to

$$
c(y)=-\frac y{|y|}\odot e^{i\kappa y/t},\qquad
P(y)=c(y)c(y)^\dagger.
\tag{NSPC.5}
$$

Its derivative constants depend explicitly on the compact
force margin, distance from zero, the kernel profiles,
$t$ and the retained finite $\kappa$. The count domain is
the literal annular condition

$$
\nu a_0r_0(|y|/t)e^{-|y|^2/(2B)}>\delta_c;
\tag{NSPC.6}
$$

the annular set before intersection with $D$ is nonempty precisely when
$\delta_c<\nu a_0r_0\sqrt B/(t\sqrt e)$.
For row normalization it is $|y|>t\delta_c/(\nu r_0)$.
The additional intersection with $D^\circ$ is retained.
At zero threshold every compact interior set avoiding
zero is allowed. A consumed calibration with a joint finite
subsequential limit only in law gives the corresponding
joint $C^2$ convergence in law, with that same retained
$\kappa$. The stronger in-probability statement is not
inferred from convergence in law. At a threshold boundary, origin,
unavailable branch or nonconvergent calibration, this
strict-domain statement makes no arbitrary extension.
:::

:::{prf:proof}
The force convergence has a uniform strict margin on
the stated neighborhood, so the original availability
test succeeds there for large $N$ with probability
tending to one. Normalize its actual nonzero force and
multiply by its actual phase. The unit-vector derivatives
through order two are bounded by the reciprocal force
margin profiles, and the phase derivatives by powers
of $|\kappa|/t$. The product and chain rules pass
the $C^2$ force convergence to its projector.
The two limit forces in (NSPC.3) are strictly negative
radial multiples of $z=y/t$, giving (NSPC.5).
Differentiating their radial magnitudes gives precisely
the available regimes in the statement. Fixed or in-probability
calibration passage uses the same chain rules. General joint
convergence in law uses continuity on each bounded calibration
core; tightness removes that core without changing the
declared convergence mode.
The actual terminal position, stage and force normalization
have not been changed.
:::

(sec-nspc-native-spatial-connection)=
## 3. Local spatial connection, frame law and noncommuting curvature

:::{prf:theorem} Actual local CSR transport has its spatial projected limit
:label: thm-nspc-local-spatial-transport

On every strict compact domain above, define from the
actual kernel projector function

$$
\omega_N=[P_N,d_yP_N],\qquad
\mathcal F_N=d_yP_N\wedge d_yP_N.
$$

For fixed or in-probability convergent calibration these
converge in $C^1$ and $C^0$, respectively, in probability to

$$
\nabla v=P\,d_y(Pv)+Q\,d_y(Qv)=d_yv+\omega v,
\qquad \omega=[P,d_yP],\qquad
\mathcal F=d_yP\wedge d_yP.
\tag{NSPC.7}
$$

This is an actual spatial
connection of the derived native line/complement splitting.
It remains reducible and adds no unrestricted gauge link law.

For a rooted actual CSR edge inside a protected bounded
scaled neighborhood, with $Y_j-Y_i=r_Nu_{ij}$,
the canonical direct rotation computed from its actual
native endpoint projectors satisfies jointly with that
same graph and its metric marks

$$
\frac{U_{j\leftarrow i}^N-I}{r_N}
 +[P_{\kappa_N}(Y_i),D_yP_{\kappa_N}(Y_i)u_{ij}]
\longrightarrow0\quad\hbox{in probability}.
\tag{NSPC.8}
$$

Here $P_{\kappa_N}$ is (NSPC.5) with the ACTUAL current
recorded phase coefficient. For fixed calibration it
is $P$; for tight joint calibrations the displayed
error still tends to zero in probability, and a joint
limit only in law yields the corresponding joint
connection/curvature/edge limit in law.
The unmarked Gaussian/Delaunay comparison now uses
$\tau=tq>0$ in Chapter NSG. There is no residual
O innovation after conditioning on terminal $Y$;
it is exactly $z=Y/t$. Protection localizes the
actual CSR edges; no replacement graph or metric
is used. Numerical graph and payload agreement
retain their separate recorded error terms.

Under a local coordinate frame $\Omega(y)\in SU(3)$,
retaining the transformed native ambient comparison,

$$
U_{j\leftarrow i}'=\Omega(Y_j)U_{j\leftarrow i}\Omega(Y_i)^\dagger,
\quad \omega'=\Omega\omega\Omega^\dagger-d\Omega\,\Omega^\dagger,
\quad \mathcal F'=\Omega\mathcal F\Omega^\dagger.
\tag{NSPC.9}
$$

This is the local transformation law of the same connection
and ambient frame. It is not a premise that arbitrary local
physical changes of native forces leave their probability
law invariant.
:::

:::{prf:proof}
The spatial $C^2$ projector convergence and polynomial
commutator formulas give the stated connection and
curvature convergence. The full projected derivative,
its anti-Hermitian traceless connection, its parallel
preservation of $P$, and the curvature formula follow
from $P^2=P$ exactly as proved in Chapter NPC.

On a bounded protected neighborhood the edge displacement
is $O(r_N)$, and the native interpolant has uniformly
bounded first and second derivatives with probability
tending to one. Taylor's formula gives

$$
P_N(Y_j)-P_N(Y_i)
=r_ND_yP_N(Y_i)u_{ij}+O(r_N^2|u_{ij}|^2).
$$

Endpoint overlap therefore tends to one, making the
canonical transport available. Its exact first-order
bound in {prf:ref}`thm-npc-native-direct-rotation`
gives $U-I=-[P_N(Y_i),P_N(Y_j)-P_N(Y_i)]+O(r_N^2)$.
Divide by $r_N$ and use the native $C^1$ convergence
to prove (NSPC.8), first uniformly on bounded $\kappa_N$
cores and then by calibration tightness. The proof never divides a pointwise
force sampling error by an uncontrolled edge length.
The native force's entire spatial derivative has converged.

For the frame law, the original ambient comparison is
$I$; in the changed coordinates it is
$\Omega(Y_j)\Omega(Y_i)^\dagger$.
Every native polar-product factor transforms at those
two endpoints, proving its exact edge law. The ambient
derivative becomes $d-d\Omega\,\Omega^\dagger$.
Applying the projected derivative to a transformed
section and using the product rule gives its displayed
connection law. Squaring that derivative gives the
curvature law. The same actual geometry and numerical
agreement transfer apply to the joint edge observation.
:::

:::{prf:theorem} Explicit noncommuting native spatial curvature
:label: thm-nspc-noncommuting-curvature

Let $R>0$ with $y=Rn$ in a strict available interior
domain, and set

$$
n=(1,2,2)/3,\qquad
a=(-4,1,1)/(3\sqrt2),\qquad b=(0,1,-1)/\sqrt2,
\qquad \beta=\kappa/t.
$$

At that actual spatial point, use its unitary phase frame
to identify the color line with $n$. The horizontal
color derivatives in the three spatial directions are

$$
h_n=i\beta\frac{2\sqrt2}{27}a,\qquad
h_a=(R^{-1}+i\beta\,10/27)a,\qquad
h_b=(R^{-1}+i\beta\,2/3)b.
\tag{NSPC.10}
$$

Consequently its rank-two complement curvature contains

$$
Q\mathcal F(n,a)Q=
\frac{2i\beta}{R}\frac{2\sqrt2}{27}aa^\dagger,
$$
$$
Q\mathcal F(a,b)Q=
K_Rab^\dagger-\overline K_Rba^\dagger,
\quad K_R=(R^{-1}+i\beta10/27)(R^{-1}-i\beta2/3).
\tag{NSPC.11}
$$

Their commutator is nonzero for every $\kappa\ne0$.
Thus this included native parameter/readout regime has
genuinely noncommuting local spatial curvature. It is
not merely a Wilson Taylor expansion or an assumed
smooth native connection. At $\kappa=0$ its curvature
is the real tangent-plane $SO(2)$ generator; all its
curvature components commute at a fixed point. That
degenerate phase regime is characterized separately.
:::

:::{prf:proof}
Remove the fixed unitary phase at the point and the
irrelevant overall sign of the unit color. For a real
spatial tangent $v$, its horizontal derivative is

$$
h_v=\frac{(I-nn^T)v}{R}
       +i\beta(I-nn^T)\operatorname{diag}(n)v.
$$

The vectors $n,a,b$ are orthonormal, and direct multiplication
gives $Q\operatorname{diag}(n)n=(2\sqrt2/27)a$,
$Q\operatorname{diag}(n)a=(10/27)a$ and
$Q\operatorname{diag}(n)b=(2/3)b$. This proves (NSPC.10).
The complement curvature is $h_uh_v^\dagger-h_vh_u^\dagger$,
so substitution gives (NSPC.11).
Its commutator is a nonzero scalar multiple of
$K_Rab^\dagger+\overline K_Rba^\dagger$ when $\beta\ne0$;
$K_R\ne0$ because both factors have real part $R^{-1}>0$.
If $\beta=0$, every horizontal derivative is real and
lies in the same two-dimensional real tangent plane.
Their skew outer products are all scalar multiples of
its single antisymmetric generator. The line curvature
vanishes and the complement curvature components commute.
These are explicit derivatives of the proved spatial
native field (NSPC.5).
:::

(sec-nspc-triangle-curvature)=
## 4. Actual native composite triangles identify the spatial curvature

:::{prf:theorem} Shrinking native CSR triangle holonomy and Wilson coefficient
:label: thm-nspc-native-triangle-curvature

In a strict available compact spatial chart, retain any
ACTUAL CSR three-cycle with root $y=Y_i$, other vertices
$Y_j=y+r_Nu$, $Y_k=y+r_Nv$, and bounded scaled vectors
$u,v$. All its endpoints, native projectors, metric marks,
face coefficients and canonical transports belong to the
same executed record. Orient its composite loop as
$i\to j\to k\to i$ and write

$$
H_{ijk}^N=U_{i\leftarrow k}^NU_{k\leftarrow j}^NU_{j\leftarrow i}^N,
\qquad W_{ijk}^N=1-\tfrac13\Re\operatorname{tr}H_{ijk}^N.
$$

On every protected bounded determining neighborhood,
uniformly over these actual three-cycles,

$$
\frac{H_{ijk}^N-I}{r_N^2}
 +\frac12\mathcal F_{\kappa_N}(y)(u,v)
\longrightarrow0\quad\hbox{in probability},
$$
$$
\frac{W_{ijk}^N}{r_N^4}
-\frac1{24}\|\mathcal F_{\kappa_N}(y)(u,v)\|_F^2
\longrightarrow0\quad\hbox{in probability}.
\tag{NSPC.17}
$$

The actual coefficient $\kappa_N$ has the calibration
scope stated above. Fixed/in-probability convergence
gives the corresponding in-probability formula with
the limiting field. A joint finite limit only in law
gives the corresponding joint field/triangle limit in law.
This is the normalized composite Wilson readout of
{prf:ref}`cor-npc-native-wilson-variation`, evaluated on
the original endpoint projectors. It does not replace
literal ray links or their face action by these matrices.
Every configured face coefficient simply multiplies
its own readout; no coefficient is chosen to enforce
an action identification.
:::

:::{prf:proof}
The same compact degree and force-margin events give
uniform third spatial derivative bounds for $P_N$:
the kernel derivatives through order three are bounded
by (NSPC.4), and the unit-force map and row quotient
are differentiated only on their actual positive margins.
Bound the retained calibration on a compact core.
This requires no third-derivative limit, only its
uniform primitive bound; calibration tightness removes
the core afterward.

For any smooth projector path over a short straight
spatial edge, let $\Delta=P_N(y+\delta)-P_N(y)$.
The exact polar identity in Chapter NPC gives

$$
U=I-[P_N(y),\Delta]-\tfrac12\Delta^2+O(|\Delta|^3).
$$

Along the same spatial edge put
$\omega=[P_N,P_N']$. The projector identities yield
$\omega^2=-(P_N')^2$ and
$\omega'=[P_N,P_N'']$.
The second-order integral expansion of its projected
parallel transport, $V'=-\omega V$, is therefore

$$
V=I-\delta_s\omega
 +\frac{\delta_s^2}{2}(\omega^2-\omega')+O(\delta_s^3),
$$

which agrees with the preceding native polar expansion
through second order. The uniform third derivatives
bound both remainders by a primitive constant times
the cubed spatial edge length.

To compute the loop coefficient explicitly, set
$A=\omega_N(y)(u)$ and $B_1=\omega_N(y)(v)$.
The three parallel edge expansions, in their native
order, are

$$
\begin{aligned}
V_{u\leftarrow0}
 &=I-r_NA+\tfrac{r_N^2}{2}(A^2-D_u\omega_u)+O(r_N^3),\\
V_{v\leftarrow u}
 &=I-r_N(B_1-A)+\tfrac{r_N^2}{2}
 [(B_1-A)^2-(D_u+D_v)(\omega_v-\omega_u)]+O(r_N^3),\\
V_{0\leftarrow v}
 &=I+r_NB_1+\tfrac{r_N^2}{2}(B_1^2+D_v\omega_v)+O(r_N^3).
\end{aligned}
$$

The first-order terms cancel. Multiplication without
commuting matrices gives the second-order sum
$-r_N^2[D_u\omega_v-D_v\omega_u+[A,B_1]]/2$.
This is $-r_N^2\mathcal F_N(y)(u,v)/2$.
Replacing each edge by its actual native direct rotation
changes the product by only $O(r_N^3)$.
The proved curvature convergence gives the first
formula of (NSPC.17).

For every unitary $3\times3$ matrix $H$ there is the
exact identity
$1-\Re\operatorname{tr}H/3=\|H-I\|_F^2/6$.
Apply it to the actual native loop and its just-proved
second-order coefficient to obtain $1/24$ in the
second formula. Protection makes the geometric vectors
bounded in the same determining neighborhood; all
metric and native face marks retain their original
correlations with that neighborhood.
:::

:::{prf:corollary} Bounded smeared composite-face readouts have their native curvature limit
:label: cor-nspc-bounded-face-readouts

Let $f$ be a fixed bounded continuous spatial test on
a strict available compact interior chart. Retain the
actual configured collection $\mathcal T_i^N$ of rooted
CSR three-cycles and every consumed face coefficient
$\beta_{ijk}^N$ of this composite readout; a literal
unit-weight readout has $\beta_{ijk}^N=1$.
Use a bounded continuous test $g$ of the ORIGINAL
scaled Wilson value, and a fixed continuous observation
cutoff $\psi$ supported on a bounded protected determining
neighborhood with at most $K_0$ rooted cycles, bounded
scaled coordinates and bounded consumed coefficients.
Its metric and coefficient limits are their actually
derived NSG estimator regimes. These observation tests
do not alter an innovation, action coefficient or edge map.
At finite $N$ retain the original available-face mask
and the readout's declared unavailable contribution.
If only available faces are defined, the expression is
the corresponding available-face submeasure. Every
face in this strict chart is available with probability
tending to one, so either convention has the same
bounded limit below; no new unavailable-face value is
assigned to the executed readout.

For fixed finite $\kappa$, the bounded assembled
original readout

$$
\frac1N\sum_i f(Y_i)\psi_i^N
 \sum_{(j,k)\in\mathcal T_i^N}\beta_{ijk}^N
                             g(r_N^{-4}W_{ijk}^N)
$$

converges in $L^2$ to

$$
\int f(x)\varphi_{tq}(x)
 E_{\Pi_{\varphi_{tq}(x)}}\left[
 \psi_*\sum_{(u,v)\in\mathcal T_*}\beta_{uv}^*
       g\left(\frac1{24}\|\mathcal F(x)(u,v)\|_F^2\right)
 \right]dx.
\tag{NSPC.18}
$$

All geometry, metrics, coefficients and vectors inside
this expectation are evaluated on the SAME Poisson
configuration and its determining adjacent stars.
Convergent in-probability calibration has its stated
limit; a tight joint calibration limit only in law
retains the corresponding law of this functional.
No unbounded action-moment passage is asserted by the
bounded-test result. Native graph/payload arithmetic
errors retain their separate NSG comparison terms.
:::

:::{prf:proof}
On the protected bounded observation domain the
triangle theorem replaces every original scaled
Wilson value by its curvature coefficient, uniformly
in probability. Uniform continuity of $g$ on its
compact coefficient range and its global boundedness
give a bounded empirical error tending to zero in
mean, hence also in $L^2$. At most $K_0$ bounded
coefficients are included; no moment of the unbounded
uncut face action is needed.

The remaining observation is a bounded fixed local
function of the original unmarked geometry with the
deterministic smooth native field (NSPC.5). Apply
{prf:ref}`thm-nga-empirical-law` with terminal Gaussian
width $tq$ and zero preparation centers. Its protected
one-neighborhood coupling retains every adjacent star,
metric and face coefficient, giving exactly (NSPC.18).
For tight calibrations, this bounded class is uniformly
continuous in $\kappa$ on every compact calibration
core. A finite calibration mesh applies the same
empirical estimate simultaneously to that class.
Removing the core preserves its actual in-probability
or joint-law calibration convergence mode. Thus no
independence between calibration and the native graph
has been inserted.
:::

(sec-nspc-source-action)=
## 5. The actual Gaussian spatial action and its source variations

:::{prf:theorem} Exact one-step spatial action and all-record weak source score
:label: thm-nspc-native-spatial-action

The original raw terminal position array has density
with the exact source action

$$
S_Y(y)=\frac1{2t^2q^2}\sum_i|y_i|^2
                +\frac{3N}{2}\log(2\pi t^2q^2).
\tag{NSPC.12}
$$

All original final velocities, force/color/projector
fields, geometry, masks and source records are their
original measurable functions or retained independent
pre-O marks. For any bounded measurable observable
$O$ of this COMPLETE output and any fixed original
O-source shift $G_i\mapsto G_i+\theta f_i$,

$$
\left.\frac d{d\theta}EO(D(tqG+\theta tqf))\right|_{\theta=0}
=E\left[O(D(tqG))\sum_i f_i\cdot G_i\right].
\tag{NSPC.13}
$$

The observation map $D$ executes the original B2 force,
cap, terminal alive test and every actual geometric
retessellation/mask branch under that shifted innovation.
It is not held fixed while the innovation changes.
The formula remains valid at their branch boundaries
because it is a weak law derivative. It uses no smooth
limit assumption about their descriptor density.
For a retained coarse connection descriptor its weak
source score is exactly
$E[\sum_i f_i\cdot G_i\mid D]$.

On a strict smooth branch with fixed calibration, the
actual source force derivatives at a native node are
fully coupled. Put $\Delta_{ij}=z_j-z_i$ and
$\delta z_i=qf_i$. In count normalization,

$$
\delta F_i=\frac\nu N\sum_{j\ne i}K_{ij}
\left[\delta z_j-\delta z_i
-\frac{t^2}{\rho^2}\Delta_{ij}
       \big(\Delta_{ij}\cdot(\delta z_j-\delta z_i)\big)\right].
\tag{NSPC.14}
$$

For row normalization use its actual
$D_i=\sum_{j\ne i}K_{ij}$: the same bracketed sum
is multiplied by $\nu/D_i$, and its derivative includes
the additional term $-F_i\delta D_i/D_i$, where
$\delta D_i=-(t^2/\rho^2)\sum_{j\ne i}K_{ij}
\Delta_{ij}\cdot(\delta z_j-\delta z_i)$.
The native color derivative is

$$
\delta c_i=\operatorname{diag}(e^{i\kappa z_i})
\left[\frac{(I-n_in_i^T)\delta F_i}{|F_i|}
                 +i\kappa n_i\odot\delta z_i\right],
\qquad n_i=F_i/|F_i|,
\tag{NSPC.15}
$$

and $\delta P_i=\delta c_ic_i^\dagger+c_i\delta c_i^\dagger$.
A varying consumed calibration adds the actual
$i\,\delta\kappa\,\operatorname{diag}(e^{i\kappa z_i})
(n_i\odot z_i)$ term. Its derivative is not omitted.
These native variations and (NSPC.13) identify the
source action and its induced spatial-connection response.
They do not identify (NSPC.12), or its coarse pushforward,
with a Yang--Mills curvature action.
:::

:::{prf:proof}
Equation (NSPC.1) gives independent centered terminal
Gaussians of variance $t^2q^2$, proving (NSPC.12).
Freeze the complete pre-O marks and use the original
Gaussian density. Changing variables in the shifted
expectation gives an integral of the bounded observation
against $p_Y(y-\theta tqf)$. Differentiation of this
Gaussian density is dominated by an integrable Gaussian
times a linear polynomial locally in $\theta$.
Its derivative is $p_Y(y)\sum_i f_i\cdot y_i/(tq)$,
proving (NSPC.13). Averaging the unchanged pre-O marks
preserves the identity. The score for a retained descriptor
is the displayed conditional expectation by the defining
property of conditional expectation. This argument
handles the entire original measurable readout, including
geometric branch changes and force-threshold boundaries.

On a strict branch, differentiate the literal Gaussian
kernel and its velocity difference to obtain (NSPC.14).
The quotient rule gives precisely the row correction.
The derivative of $F/|F|$ is $(I-nn^T)\delta F/|F|$;
the original diagonal phase derivative gives the second
term of (NSPC.15) and the stated calibration term.
Differentiate $P=cc^\dagger$ to get the projector
variation. The canonical polar-product and projected
connection derivatives are consequently those of the
same native $P$, with all simultaneous source rows kept.
No independent-force surrogate, force clipping or
unrestricted gauge-volume reference has been introduced.
:::

:::{prf:remark} Exact remaining identification and parameter boundaries
:label: rem-nspc-identification-scope

This proves a local spatial connection/curvature and its
native force/source response in an existing parameter
regime. It is a one-update native spatial limit. Multi-time
transport, stationary fluctuation dynamics, physical local
algebras and the target action remain their separately
stated obligations. The canonical connection preserves
the native line/complement splitting; it is not an
unrestricted independent $SU(3)$ gauge ensemble.

At the unchanged positive final-noise reference, Chapter
NMG instead proves nonvanishing microscopic projector
variation and failure of this same gradient scaling.
At $q=0=\sigma_x$ the consensus positions remain
coincident and its viscous colors are unavailable.
At $\nu=0$ that unavailable conclusion is exact for
every $N$. A count threshold at or above its population
radial maximum removes its limiting available region;
it does not remove every finite-sample color unless
the exact finite bound in
{prf:ref}`lem-ncf-finite-availability` also applies.
Strict final boundary, calibration, non-passive feedback,
other landscapes, graph arithmetic failures and source
deletion masks retain their actual separate branches.
In particular the literal matched B1 archive color reader
remains exactly unavailable in this consensus regime.
:::
