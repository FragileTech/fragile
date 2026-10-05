# An exact native color and geometry reflection test at positive viscosity

(sec-nic-register)=
## 1. Existing interacting algorithm and actual color instrument

:::{prf:definition} Positive-count-viscosity color-clock register
:label: def-nic-register

Retain the complete original execution record of
{prf:ref}`def-native-complete-execution-record`.
Use any finite $N\ge2$, $d=3$, harmonic force/reward with $\lambda>0$,
unbounded/all-alive boundary, original BAOAB step $h>0$,
constant isotropic independent Gaussian amplitude $b_O>0$,
friction $\gamma>0$, original positive terminal position diffusion
$\sigma_x>0$, count Gaussian viscosity $\nu,\rho>0$, and original
radial cap $C_V(w)=Vw/(V+|w|)$ with $V>0$.
There is no graph/curl/adaptive/history/elite feedback.
Choose the existing fitness exponents $p_r=p_s=0$.
Every original donor, standardizer, rescale, gate, unapplied jitter,
collision and clone-schedule parameter retains its configured value.
Constant positive fitness makes every living gate zero for every
realized update; all-alive boundary excludes revival and the
original components are singletons.

Put

$$
t=h/2,\quad c=e^{-\gamma h}\in(0,1),\quad
q^2=b_O^2(1-c^2)/(2\gamma)>0,\quad
s^2=\sigma_x^2h>0,\quad b=t(1+c),
$$
$$
t^2\lambda(1+c)=1,\quad E=1-t^2\lambda=c/(1+c)>0,\qquad
0<t\nu<E,\qquad \tau^2=t^2q^2+s^2 .
\tag{NIC.1}
$$

These are actual configured parameter branches, with a derived
strict positive-viscosity interval; neither force kick is removed.
The exact laws use the existing real-coordinate continuous-Gaussian
tag. Finite streams, rounding, numerical failures and different
variant tags retain their original comparisons.

Consume the actual spectroscopy
`ColorSource::ViscousForce { alignment: MatchedKick { B1 }, threshold }`,
its original matched clone-deletion mask and supplied fixed
`PhaseScale`/`LengthScale::Fixed`.
Its finite positive coefficient is $\kappa>0$.
The config validates finite $\delta\ge0$, INCLUDING zero.
The separate literal lecture `colors` request clamps its threshold
to $\delta_{\rm eff}\ge10^{-15}$; it only inherits the later positive
threshold test when that effective value passes.
No strictly positive threshold is silently replaced by zero.

The tested actual dictionary is the retained B1 color projector
matrix entries, availability marks and original entering positions.
It is a primitive color-and-geometry dictionary of the recorded
component-frame color, not the common-$SU(3)$ scalar orbit quotient
or an identification with that smaller quotient. Section 5 separately
tests the color-only dictionary using an actual color cylinder.
No phase-space ray or hypothetical
fermionic field is used.
Its reflection is the original update-clock link $n\mapsto1-n$,
with physical duration $t_*h>0$.
:::

(sec-nic-full-stationarity)=
## 2. Full original stationary law and Gaussian position comparison

:::{prf:theorem} Actual full stationary interacting capped phase
:label: thm-nic-full-stationarity

The original full position/velocity chain has a unique invariant
probability $\pi_{N,V}$, positive-density on the entire finite-position,
open-capped-velocity stratum.
For its stationary entering state one can use its original preceding
Gaussian sources to couple

$$
x_i=X_i^0+e_i,\qquad
X_i^0\sim N(0,\tau^2I_3)\ \hbox{independently across }i,\qquad
|e_i|\le MV,\qquad
M=b(1+2t\nu).
\tag{NIC.2}
$$

These original preceding sources are independent of the ORIGINAL
current OU sources $\xi_i\sim N(0,I_3)$.
No independence of $e_i$, current entering velocities or prepared
rows is asserted.
Every position moment is consequently bounded by its explicit
Gaussian moment plus $MV$.
:::

:::{prf:proof}
Let $L_x$ be the exact count Gaussian Laplacian, with
$0\preceq L_x\preceq I$ and off-diagonal weights $K(x_i-x_j)/N$.
Write $U=v-t\nu L_xv$.
Since $|v_i|\le V$, $|U_i|\le(1+2t\nu)V$.
Both native kicks, O, two A drifts and terminal noise give exactly

$$
z=cU-ct\lambda x+q\xi,\qquad
y=bU+tq\xi,\qquad
w=(I-t\nu L_y)z-t\lambda y,\qquad
x^+=y+s\zeta,\quad v^+=C_V(w).
\tag{NIC.3}
$$

The reset condition in (NIC.1) cancels only the position coefficient
already in the algorithm. It does not cancel either viscous force.
Equation (NIC.2) now follows from the original preceding step,
with $e=bU$ and $X^0=tq\xi_{\rm previous}+s\zeta_{\rm previous}$.
All following moment and tail comparisons use this actual coupling.

After its first update all positions are Gaussian plus a uniformly
bounded array and every velocity lies in the closed radius-$V$ ball.
These bounds make the original Cesàro laws tight at fixed $N$.
The unchanged count/cap maps are continuous, so their Feller property
and a convergent Cesàro subsequence give an invariant probability.
Every transition puts zero mass on the cap boundary, so the invariant
law is supported in the open velocity ball.

For completeness its physical transition has positive density almost
everywhere. Conditional on A1 preparation $p$ and B1 velocity, its
uncapped B2 map of $z$ is
$K_p(z)=Ez-t\lambda p-t\nu L_{p+tz}z$.
The bound $0\preceq L\preceq I$ makes this map proper under
$E-t\nu>0$. Homotopy from zero viscosity has degree one and
therefore a preimage for every target.
Its analytic Jacobian determinant is nonzero at $z=0$:
there it is $\det(EI-t\nu L_p)>0$.
Its critical-source set is consequently Lebesgue-null.
Regular targets outside the Sard-null critical-value set have
preimages with nonzero determinant and strictly positive original
Gaussian source density. Their uncapped output density is positive.
The original final Gaussian position map has nonzero scale $s$;
the original cap is a $C^1$ diffeomorphism onto its open ball.
The joint physical transition therefore has positive density almost
everywhere, by its actual triangular source Jacobian.

This also proves uniqueness without an assumed invariant relative law.
If two invariant probabilities existed, their mean invariant law
$\rho$ has positive density. For $f=d\pi_1/d\rho\le2$,
invariance gives $f(S_1)=E[f(S_0)\mid S_1]$.
Stationarity and equality of its two second moments imply zero
conditional variance, so $f(S_0)=f(S_1)$ almost surely.
The strictly positive joint transition density makes $f$ constant
almost everywhere. Thus $\pi_1=\pi_2$.
:::

(sec-nic-primitive-reflection)=
## 3. Primitive same-law antisymmetric color correlation

:::{prf:definition} Fully evaluated reflection constants
:label: def-nic-reflection-constants

Use only original parameters in (NIC.1) and define

$$
\begin{gathered}
A_v=1+2t\nu,\quad M=bA_v,\quad \ell_\rho=e^{-1/2}/\rho,\\
Z_e=cA_v+ct\lambda M,\quad
A_e=Z_e+t\lambda M,\quad B_e=2t\nu M\ell_\rho,\\
Z_4=15^{1/4}\sqrt{(ct\lambda\tau)^2+q^2},\\
\sigma_-=\tau ct\lambda(1-t\nu)>0,\quad
\sigma_+=\tau ct\lambda,\quad
M_\xi=q\sqrt3(E+2t\nu),\\
g_*=\tau^2ct\lambda(1-t\nu)\,
 \pi\sigma_-^2(2\pi\sigma_+^2)^{-3/2}
 \exp[-(\sigma_-+2M_\xi)^2/(2\sigma_-^2)]>0,\\
C_*=
 M+\frac{\tau}{\sigma_-}
       [1+2\,3^{1/4}(A_e+2B_eZ_4)],\\
V_*=\min\{1,\pi/(4\kappa),g_*/[2(C_*+M)]\}>0 .
\end{gathered}
\tag{NIC.4}
$$

The complete original noise remains unbounded.
No coefficient depends on $N$ or an unknown stationary density.
:::

:::{prf:theorem} Actual positive-viscosity color-and-geometry clock test
:label: thm-nic-color-clock

For every $N\ge2$ and ORIGINAL cap $0<V<V_*$,
the actual spectroscopy threshold-$0$ color-and-geometry history
fails its literal adjacent-update reflection positivity.
Its actual bounded projector readout recovers
$a\cdot v_i$, where $a=(e_1-e_2)/\sqrt2$.
The original finite-moment pair obeys the uniform strict estimate

$$
J_V=
 E_{\pi_{N,V}}[(a\cdot x_i)(a\cdot v_i^+)]
 -E_{\pi_{N,V}}[(a\cdot v_i)(a\cdot x_i^+)]
 \le-\tfrac12g_*V<0 .
\tag{NIC.5}
$$

Clipping the OBSERVATION $a\cdot x_i$ at a sufficiently large
explicit radius preserves this antisymmetry. No algorithm state,
innovation or force is clipped.
:::

:::{prf:proof}
The actual B1 input is the entering $v$, because every gate was
proved zero. Its force is $F=-\nu L_xv$.
For fixed positions its own diagonal coefficient is nonzero:
$D_{ii}=N^{-1}\sum_{j\ne i}K(x_i-x_j)>0$.
Absolute continuity of the derived invariant law makes each
$F_{ia}=0$ a null hyperplane, after conditioning on positions.
Thus the threshold-zero color is available and $F_{i1}F_{i2}\ne0$
almost surely. Since $2\kappa V<\pi/2$, its ORIGINAL entry

$$
(P_i)_{12}=\frac{F_{i1}F_{i2}}{|F_i|^2}
                     e^{i\kappa(v_{i1}-v_{i2})}
$$

gives the exact measurable recovery

$$
a\cdot v_i=\frac1{\sqrt2\kappa}
 \arctan\frac{\Im(P_i)_{12}}{\Re(P_i)_{12}} .
\tag{NIC.6}
$$

The sign of the real force product does not affect this ratio;
the original phase interval selects the correct arctangent.
This uses actual color matrix entries, not a new velocity coordinate.

Use (NIC.2), $y^0=tq\xi$, $z^0=-ct\lambda X^0+q\xi$, and the
actual comparison variable

$$
w^0=-ct\lambda(I-t\nu L_{y^0})X^0
                   +q(EI-t\nu L_{y^0})\xi .
\tag{NIC.7}
$$

Conditional on current $\xi$, its $i$th row is a three-dimensional
isotropic Gaussian. Its scalar variance lies between
$\sigma_-^2$ and $\sigma_+^2$.
Its mean has $L^2$ norm at most $M_\xi$, using the original
count row sum, $|K|\le1$ and the Gaussian norm moment $\sqrt3$.
Hence $|\mathrm{mean}(w_i^0\mid\xi)|\le2M_\xi$ with probability
at least $3/4$.
Gaussian integration by parts in the ORIGINAL independent $X^0$
gives

$$
E[(a\cdot X_i^0)(a\cdot w_i^0/|w_i^0|)]
=-\tau^2ct\lambda E\!\left[
 (1-t\nu D_{ii}(y^0))
 \frac{|w_i^0|^2-(a\cdot w_i^0)^2}{|w_i^0|^3}\right]
\le-g_* .
\tag{NIC.8}
$$

To verify the last fully primitive inequality, on that mean event
the conditional density on $|w|\le\sigma_-$ is at least
$(2\pi\sigma_+^2)^{-3/2}
e^{-(\sigma_-+2M_\xi)^2/(2\sigma_-^2)}$.
The integral of $(1-(a\cdot w/|w|)^2)/|w|$ on this ball is
$4\pi\sigma_-^2/3$.
Multiply by the event probability $3/4$ and
$1-t\nu D_{ii}\ge1-t\nu$ to get exactly $g_*$.
One may first regularize the unit vector by
$w/(|w|^2+\eta^2)^{1/2}$; the original Gaussian inverse moments
justify integration by parts and passage to its unit-vector limit.
All singularities in this integration are locally integrable in
three dimensions.

The original same-record kernel perturbation obeys

$$
|w_i-w_i^0|
\le V[A_e+B_e(|z_i^0|+\overline{|z^0|})],
\qquad
\|w_i-w_i^0\|_4\le V(A_e+2B_eZ_4).
\tag{NIC.9}
$$

Indeed $|z_i-z_i^0|\le Z_eV$ and $|y_i-y_i^0|\le MV$.
The matrix $I-t\nu L_y$ has nonnegative row entries and row sum
one. Its action on the first difference is bounded by $Z_eV$.
Differentiating the actual kernel costs at most
$2MV\ell_\rho$ per pair; its full original count sum gives the
$B_e$ term. The potential difference costs $t\lambda MV$.
The original Gaussian $z_i^0$ has fourth-norm $Z_4$,
and Minkowski gives the same bound for its row average.
No correlated product is factorized.

For $g_V(w)=w/(V+|w|)$,
$|g_V(w)-w^0/|w^0||\le(2|w-w^0|+V)/|w^0|$.
Every conditional Gaussian above has
$E|w_i^0|^{-2}\le\sigma_-^{-2}$.
For example its Laplace representation
$|w|^{-2}=\int_0^\infty e^{-u|w|^2}du$ shows that a mean shift
can only decrease this moment, and the centered three-Gaussian
value is its reciprocal variance.
Hölder with exponents $(4,4,2)$, the original fourth moment
$\|a\cdot X_i^0\|_4=3^{1/4}\tau$ and (NIC.9) therefore gives

$$
\left|E[(a\cdot x_i)(a\cdot v_i^+/V)]
 -E[(a\cdot X_i^0)(a\cdot w_i^0/|w_i^0|)]\right|
\le C_*V .
$$

For the other covariance, current independent centered $\xi,\zeta$
in (NIC.3) give
$|E[(a\cdot v_i/V)(a\cdot x_i^+)]|\le MV$.
Combine this, (NIC.8) and the cap test to prove (NIC.5).

Finally the position comparison gives, for $R>MV$,

$$
E[|a\cdot x_i|\mathbf1_{\{|a\cdot x_i|>R\}}]
\le[\tau\sqrt{2/\pi}+2MV]\,
 e^{-(R-MV)^2/(2\tau^2)} .
\tag{NIC.10}
$$

It follows by conditioning on the original one-dimensional
Gaussian $a\cdot X_i^0$ and using its exact absolute tail moment.
Clipping just this recorded observation changes $J_V$ by at most
twice $V$ times (NIC.10); choose that bound below $g_*V/4$.
The bounded real tests $O_1=\operatorname{clip}_R(a\cdot x_i)$,
$O_2=a\cdot v_i$ are functions of the actual dictionary.
Their reflected pair matrix is not Hermitian, by (NIC.5).
Explicitly the adjacent reflection form of the actual cylinder
$O_1+iO_2$ has nonzero imaginary part
$E[O_1O_2^+]-E[O_2O_1^+]$.
A positive sesquilinear form must be Hermitian and have real
nonnegative diagonals. The native clock reflection therefore fails.
:::

(sec-nic-positive-threshold)=
## 4. The actual strictly positive threshold regime

:::{prf:theorem} Primitive finite-population availability and threshold comparison
:label: thm-nic-positive-threshold

Let $T_0=\tau+M$ and choose
$0<\epsilon\le(g_*/(64T_0))^2$.
For an actual cap $0<V<V_*$ choose finite analysis radii
$R_x>MV$, $W>A_wV$, where

$$
 A_w=(1+2t\nu)Z_e+t\lambda M,\quad
 C_w=(1+2t\nu)ct\lambda\tau+
                    q(1+2t\nu+t^2\lambda).
$$

Require the explicit original Gaussian tail tests

$$
N2^{3/2}e^{-(R_x-MV)^2/(4\tau^2)}\le\epsilon,\qquad
2N2^{3/2}e^{-(W-A_wV)^2/(4C_w^2)}\le\epsilon .
\tag{NIC.11}
$$

Retain the actual B2 target-local inverse condition of
{prf:ref}`thm-nyc-uniform-target-inverse`:

$$
\begin{gathered}
E_W=\frac{W+t^2\lambda\nu\rho/\sqrt e}{1-2t\nu},\quad
C_W=1+2t^2\lambda/e+2tE_W/(\rho\sqrt e),\\
2t\nu<1,\qquad \beta_W=E-2t\nu C_W>0 .
\end{gathered}
\tag{NIC.12}
$$

Put $U=VW/(V+W)$, $\kappa_x=e^{-2R_x^2/\rho^2}$,
$v_3=4\pi/3$, and evaluate

$$
\begin{gathered}
D_*=(2\pi qs)^{-3N}\beta_W^{-3N}(1+W/V)^{4N},\\
K_*=
D_*(v_3R_x^3)^N(v_3U^3)^{N-1}v_3
       \left[\frac{N}{\nu(N-1)\kappa_x}\right]^3,\qquad
\delta_*=(\epsilon/K_*)^{1/3}>0 .
\end{gathered}
\tag{NIC.13}
$$

For every ORIGINAL effective threshold $0<\delta\le\delta_*$,
the actual spectroscopy color-and-geometry history still fails
its literal adjacent-update reflection test.
The literal lecture's clamped threshold only passes this
certificate if its actual $\delta_{\rm eff}\le\delta_*$.
If a primitive test fails, no reflection conclusion for that
different threshold is assigned.
:::

:::{prf:proof}
The original position coupling gives the first tail in (NIC.11).
From (NIC.3),
$\max_i|w_i|\le A_wV+
C_w\max_i\{|G_i^{\rm previous}|,|\xi_i|\}$,
where the normalized previous combination $X_i^0/\tau$ is an
original standard Gaussian. Exponential Markov with
$Ee^{|G_3|^2/4}=2^{3/2}$ and union over these $2N$ rows gives
the second tail. No innovation is truncated.

On $\max|x_i|\le R_x$, $\max|v_i|\le U$, the original stationary
joint density is at most $D_*$ by the target-local B2 Jacobian
bound, cap inverse Jacobian and original final Gaussian scale.
This bound is preparation independent, so integrating the actual
invariant entering law preserves it.
The entering count force has
$D_{ii}\ge(N-1)\kappa_x/N$.
Conditioning on positions and the other velocities, its
$|F_i|\le\delta$ set is a ball for $v_i$ of radius at most
$N\delta/[\nu(N-1)\kappa_x]$.
Integrate this ball against the FULL joint density bound and
the displayed finite position/other-velocity volumes. The result is

$$
P_{\pi_{N,V}}(|F_i|\le\delta)\le2\epsilon+K_*\delta^3
                                      \le3\epsilon .
\tag{NIC.14}
$$

Define the actual available-row color observation by (NIC.6)
on its available rows and zero otherwise. It differs from the
true $a\cdot v_i$ by at most $V$ only on that original unavailable
event. Cauchy--Schwarz and $\|a\cdot x_i\|_2\le\tau+MV\le T_0$
show that its antisymmetric correlation differs from (NIC.5)
by at most $2VT_0\sqrt{3\epsilon}<g_*V/16$.
Thus the actual positive-threshold pair remains asymmetric.
Choose the observation-only position truncation from (NIC.10)
to cost less than $g_*V/16$ as well.
The same bounded complex cylinder proves reflection failure.
Every unavailability probability is paid; no normalized force
direction is assumed available merely because it is nonzero.
:::

(sec-nic-color-only)=
## 5. An actual negative reflected diagonal in the color-only algebra

:::{prf:definition} Primitive color-only error budget
:label: def-nic-color-only-budget

In the same unchanged algorithm define

$$
\begin{gathered}
\sigma_W^2=(ct\lambda\tau)^2+q^2E^2,\qquad
r=-\frac{ct^2\lambda q^2E}{\sigma_W^2}\in(-1/2,0),\\
g_{\rm col}=\frac{8|r|}{9\pi}>0,\qquad
K_{\rm col}=1+2A_e+4B_eZ_4,\\
V_{\rm col}=\min\{1,\pi/(4\kappa),
                         \sigma_Wg_{\rm col}/(8K_{\rm col})\}.
\end{gathered}
\tag{NIC.15}
$$

All constants are functions of the original parameters. Impose
the explicitly evaluated positive-viscosity/cap regime

$$
0<t\nu<\min\{E,\sigma_Wg_{\rm col}/(32Z_4)\},
\qquad 0<V<V_{\rm col}.
\tag{NIC.16}
$$

This is a nonempty original parameter interval: at fixed
$h,\gamma,b_O,\sigma_x,\rho,\kappa$, every constant except
the displayed original $t\nu$ is positive and continuous at
$\nu=0$, and the right-hand viscosity bound is itself independent
of $\nu$. Both original viscous kicks remain present for
every $\nu>0$ in the interval.
:::

:::{prf:theorem} Full recorded viscous-color history fails the original reflection
:label: thm-nic-color-only-reflection

For every finite $N\ge2$ and (NIC.16), the actual threshold-zero
viscous-color PROJECTOR history fails reflection positivity
at its original adjacent-update clock. This concerns the
COLOR-ONLY algebra: no geometry coordinate is consumed.
Its actual bounded real projector cylinder

$$
O_i(P)=\frac1{V\sqrt2\kappa}
 \arctan\frac{\Im P_{i,12}}{\Re P_{i,12}}
       =\frac{a\cdot v_i}{V},
\qquad |O_i|\le1,
$$

obeys the strict reflected diagonal estimate

$$
E_{\pi_{N,V}}[O_i(P_n)O_i(P_{n+1})]
                 \le-\tfrac12g_{\rm col}<0 .
\tag{NIC.17}
$$

For an original positive threshold, replace $O_i$ by its actual
available-row value and zero on unavailable rows.
Choose
$\epsilon_{\rm col}=g_{\rm col}/48$ and evaluate the ORIGINAL
Gaussian tail/Jacobian tests and constants (NIC.11)--(NIC.13)
with $\epsilon_{\rm col}$ in place of $\epsilon$.
Whenever those tests pass and $0<\delta\le\delta_*$, its actual
color-only reflected diagonal is at most $-g_{\rm col}/4$.
The separate clamped lecture instrument inherits this test
only if its actual $\delta_{\rm eff}$ passes.
:::

:::{prf:proof}
Use a stationary original chain with three successive independent
OU source lists and the original independent terminal Gaussian
lists. Its SAME preceding-source comparison in (NIC.2) gives,
for each row, the explicit Gaussian array

$$
\overline W_{i,n}
 =-ct\lambda(tq\xi_{i,n-1}+s\zeta_{i,n-1})
                             +qE\xi_{i,n}.
\tag{NIC.18}
$$

Its variance is $\sigma_W^2I_3$, and its adjacent cross-covariance
is $-ct^2\lambda q^2E\,I_3=r\sigma_W^2I_3$.
The term $s^2>0$ makes $|r|<1/2$ by
$A^2+B^2\ge2AB$. These are the ORIGINAL executed noises,
not a replacement process for the native chain.

For standard three-Gaussians $G,H$ with
$E[GH^T]=rI_3$, put $f(G)=a\cdot G/|G|$.
Its normalized Hermite expansion has only odd total degrees,
because $f(-G)=-f(G)$.
The native Gaussian Hermite calculation gives

$$
E[f(G)f(H)]
 =\sum_{\ell\ {\rm odd}}r^\ell
                  \|\operatorname{Proj}_\ell f\|_2^2
 \le r\|\operatorname{Proj}_1f\|_2^2
 =\frac{8r}{9\pi}=-g_{\rm col}.
\tag{NIC.19}
$$

Indeed rotational invariance gives
$E[G_a f(G)]=E|G|/3=2\sqrt{2/\pi}/3$,
and the other first-degree coefficients are zero.
The $L^2$ expansion converges absolutely for this $|r|<1$.
Every odd-degree summand has the same negative sign;
none is dropped with the wrong inequality.

To compare to the ACTUAL velocities, (NIC.9) and
$w_i^0-\overline W_i=-t\nu(L_{y^0}z^0)_i$ give

$$
\|w_i-\overline W_i\|_2
 \le V(A_e+2B_eZ_4)+2t\nu Z_4 .
$$

The full correlated count sum is retained: pointwise
$|(L_{y^0}z^0)_i|\le|z_i^0|+\overline{|z^0|}$.
Using the earlier cap comparison, the original centered
Gaussian inverse moment
$E|\overline W_i|^{-2}=\sigma_W^{-2}$, and Cauchy--Schwarz
gives the actual one-row error

$$
E\left|\frac{v_{i,n+1}}V
              -\frac{\overline W_{i,n}}{|\overline W_{i,n}|}\right|
 \le \epsilon_{\rm dir}:=
       \frac{VK_{\rm col}+4t\nu Z_4}{\sigma_W}.
\tag{NIC.20}
$$

No error is made independent of its inverse Gaussian radius.
The same bound applies to the preceding update by stationarity.
All compared normalized vectors have norm at most one.
Thus the difference between their two adjacent scalar-product
expectations is at most $2\epsilon_{\rm dir}$.
The two tests in (NIC.16) give
$2\epsilon_{\rm dir}<g_{\rm col}/2$.
Equations (NIC.19)--(NIC.20) prove (NIC.17), using exactly
the actual phase recovery of (NIC.6).
Its negative real reflected diagonal is impossible for a
positive reflection form.

For a strictly positive threshold, the availability part of
the proof of (NIC.14) uses only $V\le1$, the original tail
tests and the target-local Jacobian test; it does not use
the color-and-geometry truncation or its special choice
of $\epsilon$. It therefore gives
$P(|F_i|\le\delta)\le3\epsilon_{\rm col}=g_{\rm col}/16$.
Changing a normalized scalar color observation to zero on
that event changes its adjacent product expectation by at most
$2P(|F_i|\le\delta)\le g_{\rm col}/8$.
The actual positive-threshold reflected diagonal is hence
at most $-3g_{\rm col}/8$, which implies the stated weaker
$-g_{\rm col}/4$ bound. Every original availability mark is kept.
:::

:::{prf:example} Explicit finite positive-viscosity and positive-threshold witness
:label: ex-nic-witness

Take the EXISTING real-coordinate count algorithm with
$N=3$, $h=\gamma=q=s=\kappa=1$, $\rho=100$ and $\nu=10^{-6}$.
Set its original OU amplitude and confining force to

$$
b_O=\sqrt{2/(1-e^{-2})},\qquad
\lambda=\frac4{1+e^{-1}},
$$

and original fixed phase scales to mass $=h_{\rm eff}=$ length $=1$.
Every source remains its original unbounded Gaussian.
The primitive color-and-geometry constants evaluate to

$$
\begin{gathered}
E=.2689414213\ldots,\quad \tau=1.1180339887\ldots,\\
g_*=.00865407097\ldots,\quad
C_*=11.03709034\ldots,\quad V_*=.0003691685128\ldots .
\end{gathered}
$$

Choose the ORIGINAL cap $V=V_*/2=.0001845842564\ldots$.
The Gaussian color-only correlation has $r=-1/6$ and
$g_{\rm col}=4/(27\pi)=.04715702017\ldots$.
It passes (NIC.16).
For a single common availability certificate choose
$\epsilon=(g_*/(64(\tau+M)))^2=5.6309766610\cdot10^{-9}$
and the equality radii in (NIC.11):

$$
R_x=10.27955132\ldots,\quad W=21.79400058\ldots,\quad
\beta_W=.2689397512\ldots,\quad
\delta_*=4.4398064512\cdot10^{-26}>0 .
$$

This $\epsilon$ is smaller than $\epsilon_{\rm col}$, so both
actual dictionaries pass their positive-threshold tests for
every original spectroscopy $0<\delta\le\delta_*$.
For example its permitted original $\delta=10^{-26}$ passes.
The literal clamped lecture threshold does not pass THIS
conservative certificate, because $10^{-15}>\delta_*$;
no conclusion about that different instrument follows from it.
:::

:::{prf:remark} Exact instrument and physical-sector scope
:label: rem-nic-scope

This is an interacting $\nu>0$ theorem for an actual full
stationary native law, actual B1 viscous colors, actual geometry
and unchanged unbounded sources. Its cap/threshold/phase regimes
are given by primitive formulas, including strictly positive
thresholds that pass (NIC.11)--(NIC.13).
At fixed positive threshold and $V\downarrow0$, force availability
cannot be inferred from the threshold-zero theorem: indeed
$|F_i|\le2\nu V$ makes every row unavailable once
$2\nu V\le\delta$. That is a DIFFERENT actual parameter regime.

The first test concerns the original component-frame color PROJECTOR
and position dictionary. Section 5 gives its separate direct
color-only test, with its own evaluated viscosity/cap/availability
regime. Neither is a test of the smaller common-$SU(3)$ scalar orbit;
no geometry mode is invented as a color-only observable.
Different recording strides/reflections, an actually derived
physical subalgebra, continuum defect disappearance, and a
positive interacting color-only reconstruction retain their
own native obligations. The positive NTC/NHR families and their
proved gaps are not negated by this failed literal clock test.
:::
