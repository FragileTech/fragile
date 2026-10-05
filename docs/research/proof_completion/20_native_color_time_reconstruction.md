# Actual color/geometry clock reflection in an interacting native regime

(sec-nct-complete-record)=
## 1. The executed unbounded count branch

:::{prf:definition} Native color-history execution and observation register
:label: def-nct-complete-record

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`. The present positive-
viscosity calculation uses the existing real-coordinate BAOAB count
variant with quadratic provider $F(x)=-\lambda x$, unbounded boundary,
`velocity_cap=None`, passive observations, period-one cloning and
$p_r=p_s=0$. The last choice is an existing fitness exponent convention:
its actual positive fitness is constant, so every original accepted
clone probability is zero. Every donor width, measurement regularizer,
standardizer, feature clamp, rescale, acceptance scale and floor, self
exclusion, jitter coefficient and component collision coefficient retains
its configured value. There is no mandatory revival because this boundary
and the all-alive initial stratum have no deaths. The original zero-
accepted-edge components are singletons, whose collision map is exactly
the identity. No donor history, elite, curl or consumed metric feedback
is enabled. This is a declared parameter regime of the existing update,
not a replacement of the capped terminal-box reference.

Write

$$
a=h/2>0,\quad c=e^{-\gamma h}\in(0,1),\quad
q^2=\frac{b_O^2(1-c^2)}{2\gamma}>0,\quad
s^2=\sigma_x^2h>0,\quad E=1-a^2\lambda,
$$
$$
p=1-a^2\lambda(1+c),\quad b=a(1+c),\quad
D=c-a^2\lambda(1+c),\quad k=-a\lambda(c+p),\quad
A=\begin{pmatrix}p&b\\k&D\end{pmatrix}.
\tag{NCT.1}
$$

The configured count Gaussian kernel has its original width $\rho>0$.
Its actual Laplacian is symmetric, $L_x\mathbf1=0$ and
$0\preceq L_x\preceq I$ by {prf:ref}`lem-cgd-count-kick`.
All noises remain their original independent unbounded Gaussians.
The analytic laws below have the real-arithmetic convention in this
complete record. Fixed streams and floating arithmetic retain their
separately recorded laws; exact cancellation is not asserted for rounded
sums. A row-normalized or feedback variant does not inherit the count
centroid identities.

For the executed common-$SU(3)$ color channel take $d=3$; the Lyapunov,
centroid and geometry calculations themselves hold in any finite $d$.
The observation algebra retains the original matched B2 common-$SU(3)$
color orbit, original force threshold $\delta_c\ge0$, phase coefficient
$\kappa$, masks and terminal positions, together with its complete
stationary trajectory cylinders. Absolute event/version/step labels align
those observations with their executed updates; they are not added as
an ever-increasing random coordinate of a stationary core law.
Native clock reflection is $n\mapsto-n$,
with duration $a_{\rm phys}=t_*h>0$ per step and position calibration
$x_*>0$. It is the native sampling-clock identification being tested.
The separately declared four-position-coordinate physical cut keeps its
own embedding and is not identified with this clock.
:::

(sec-nct-full-invariant-law)=
## 2. A primitive full-law stationary witness

:::{prf:definition} Explicit quadratic Lyapunov matrix and viscosity interval
:label: def-nct-lyapunov-budget

Assume $0<a^2\lambda<1$. Then $\det A=c$ and
$\operatorname{tr}A=(1+c)(1-2a^2\lambda)$ imply that both eigenvalues of
$A$ have modulus less than one. For a conjugate pair their common modulus
is $\sqrt c<1$. For real roots, the positive values of its characteristic
polynomial at $1$ and $-1$, together with product $c\in(0,1)$, exclude
roots beyond either endpoint.

Define the two-by-two matrix $H$ by the explicit primitive linear system

$$
\begin{pmatrix}
1-p^2&-2pk&-k^2\\
-pb&1-pD-bk&-kD\\
-b^2&-2bD&1-D^2
\end{pmatrix}
\begin{pmatrix}H_{11}\\H_{12}\\H_{22}\end{pmatrix}
=\begin{pmatrix}1\\0\\1\end{pmatrix}.
\tag{NCT.2}
$$

Equivalently $H=\sum_{j\ge0}(A^j)^{\mathsf T}A^j$.
The proved stability makes that series convergent, positive definite and
at least $I$; it proves both invertibility of (NCT.2) and
$H-A^{\mathsf T}HA=I$. Put

$$
\begin{gathered}
h_-=\lambda_{\min}(H),\quad h_+=\lambda_{\max}(H),\quad
r_0=\sqrt{1-h_+^{-1}},\\
K_0=a\left[\sqrt{b^2+D^2}
                 +c\sqrt{(a\lambda)^2+4}\right],\\
\nu_{\rm L}=\min\left\{\frac1a,\frac E{2a},
 \frac{1-r_0}{2K_0\sqrt{h_+/h_-}}\right\}>0.
\end{gathered}
\tag{NCT.3}
$$

In the following statements $0<\nu\le\nu_{\rm L}$. No invariant-law
hypothesis is included in these primitive tests. All parameters are fixed
independently of population. Define

$$
\|S\|_{H,N}^2=\frac1N\sum_i
\begin{pmatrix}x_i\\v_i\end{pmatrix}^{\mathsf T}
(H\otimes I_d)\begin{pmatrix}x_i\\v_i\end{pmatrix},\qquad
r=r_0+\nu K_0\sqrt{h_+/h_-}<1,
$$
$$
K_\xi=q\left[\sqrt{(a,E)H(a,E)^{\mathsf T}}
                       +a\nu\sqrt{h_+}\right],\qquad
K_\zeta=s\sqrt{H_{11}}.
\tag{NCT.4}
$$
:::

:::{prf:theorem} The full interacting native law exists, is unique and has all polynomial moments
:label: thm-nct-full-native-invariant

For every permitted finite $N$ in the preceding complete count regime,
the actual conservative position/velocity kernel has a unique invariant
probability $\pi_N$. This is the full interacting law; no invariant
relative-coordinate law is assumed. It has, for every $u\ge1$, the explicit
moment bound

$$
\left(\mathbb E_{\pi_N}\|S\|_{H,N}^u\right)^{1/u}
\le\frac{(K_\xi+K_\zeta)g_{Nd,u}/\sqrt N}{1-r},
\quad g_{m,u}=\left(\mathbb E|Z_m|^u\right)^{1/u}.
\tag{NCT.5}
$$

Thus no confining support cutoff or bounded innovation replaces the actual
unbounded algorithm. The law has full support and is equivalent to
Lebesgue measure on the entire position/velocity space. Its actual
transition-attached stationary color/geometry record law exists by
integrating the original passive record kernel against this $\pi_N$.
:::

:::{prf:proof}
For the actual first kick, $\delta_1=-a\nu L_xv$ has norm at most
$a\nu\|v\|_{2,N}$. Relative to the zero-viscosity affine map,
its change in A2 positions is $b\delta_1$ and its change in the linear
part of the second velocity kick is $D\delta_1$. The actual last viscous
kick has norm at most

$$
\|\delta_2\|_{2,N}\le a\nu\|z\|_{2,N}
\le a\nu\{c[a\lambda\|x\|_{2,N}
                   +(1+a\nu)\|v\|_{2,N}]
                         +q\|\xi\|_{2,N}\}.
$$

Since $a\nu\le1$, the combined state error obeys

$$
\|\Delta\|_{2,N}\le\nu K_0\|S\|_{2,N}
                        +a\nu q\|\xi\|_{2,N}.
$$

The quadratic Lyapunov identity gives
$\|AS\|_{H,N}\le r_0\|S\|_{H,N}$. Its true Gaussian affine noise is
$(aq\xi+s\zeta,Eq\xi)$. The preceding error bound, including the
O-dependent last viscous force, proves the pathwise inequality

$$
\|S'\|_{H,N}\le r\|S\|_{H,N}
             +K_\xi\|\xi\|_{2,N}+K_\zeta\|\zeta\|_{2,N}.
\tag{NCT.6}
$$

No independence between $\delta_2$ and $\xi$ is used. The original
independent Gaussian arrays have
$\|\xi\|_{2,N}=|Z_{Nd}|/\sqrt N$, with the same formula for $\zeta$.
Start the actual kernel from zero. Iterated Minkowski in (NCT.6) bounds
its $u$-moment by the right side of (NCT.5), uniformly in time. In
particular the averaged time marginals are tight in the full finite-
dimensional state space. The exact update map is continuous in its input
for fixed innovations, so dominated convergence makes its kernel Feller.
A weakly convergent subsequence of those averages is invariant: for a
bounded continuous $f$, the difference between its average after one
update and before that update is the first/last telescoping difference
divided by the number of terms. Feller continuity passes this identity to
the weak limit. Lower semicontinuity transfers each uniform moment bound
and proves (NCT.5) for that invariant law.

For uniqueness, $E-a\nu>0$, $q,s>0$ and the actual nonlinear phase
smoothing theorem apply on the entire entering space. Its proper
surjective analytic map has a null critical source set. On countably many
bounded source pieces it is Lipschitz, so the image of that null set is
also null. Every other target has a regular preimage, and change of
variables of the strictly positive Gaussian density yields a strictly
positive transition density there. Thus each actual row has a density
positive almost everywhere. Tonelli shows that any invariant probability
has a density positive almost everywhere; all invariant probabilities are
equivalent to Lebesgue measure.

If $\pi_1,\pi_2$ were invariant, put
$\pi=(\pi_1+\pi_2)/2$ and $f=d\pi_1/d\pi$, so $0\le f\le2$.
Invariance makes $f$ fixed by the reverse kernel relative to $\pi$.
Conditional Jensen has equality in its squared-norm contraction, because
that function is fixed. Its conditional variance is therefore zero.
Almost every reverse row is equivalent to $\pi$, by the positive density
just proved, so $f$ is constant $\pi$-almost surely. Normalization gives
$f=1$ and $\pi_1=\pi_2$. This proves uniqueness without postulating a
stationary mixing or relative-coordinate law. Finally attach the original
passive update record to this invariant core law; its incoming record is
not consumed, so the resulting joint law is stationary for the actual
augmented kernel.
:::

(sec-nct-centroid-transfer)=
## 3. The exact centroid sector of this full native law

:::{prf:theorem} Native interacting centroid correspondence and its linear transfer
:label: thm-nct-native-centroid-sector

For the law just constructed, the normalized centroid
$Z_n=\sqrt N(\overline x_n,\overline v_n)$ obeys the exact stationary
Gaussian chain

$$
Z_{n+1}=AZ_n+\eta_n,\qquad
\operatorname{Cov}(\eta_n)=
\Sigma\otimes I_d,\quad
\Sigma=\begin{pmatrix}a^2q^2+s^2&aq^2E\\
aq^2E&q^2E^2\end{pmatrix}.
\tag{NCT.7}
$$

Its primitive stationary covariance is
$C=\sum_{j\ge0}A^j\Sigma(A^j)^{\mathsf T}$.
The full core law factors into this centroid law and the actual invariant
relative law. Positive viscosity acts on the latter; it is not removed
from the algorithm to obtain this identity.

Centroid functions form a genuine reducing subspace of the full native
transfer: their invariant injection is the composition with the actual
centroid, and its adjoint is conditional expectation. On the linear
centroid functions the transfer is the explicitly computed matrix $A$,
up to the covariance Gram normalization $C$. Those eigenvalues may be a
complex conjugate pair or negative real values, despite their modulus
being less than one. No self-adjoint positive physical Hamiltonian is
inferred from their native decay.
:::

:::{prf:proof}
Both actual count viscous forces sum to zero and depend only on relative
positions and velocities. The quadratic force has mean
$-\lambda\overline x$. The original centered Gaussian arrays split into
independent centroid and relative arrays. Applying the actual two kicks,
O stage and position innovation therefore gives (NCT.7), while its full
relative update depends only on relative entering coordinates and relative
innovations. Their joint core transition is a product kernel. The proved
full invariant law has stationary marginals for both kernels. Their
product is also invariant, so full-law uniqueness proves the asserted
factorization. This supplies, rather than assumes, the actual relative
stationary law. The positive-definite noise covariance and stability of
$A$ give the stated Gaussian centroid covariance by its convergent
innovation series.

Composition with the actual centroid is an isometry for this pushforward
law. Product factorization makes its orthogonal projection commute with
the full transition and its adjoint: conditional expectation over the
stationary relative coordinate before or after its update agrees. Thus it
is a genuine reducing sector. For a linear observable $u^{\mathsf T}Z$,
the next conditional mean is $u^{\mathsf T}AZ$; this proves the linear
matrix and its Gram normalization. The eigenvalue statement follows from
the characteristic polynomial of $A$.
:::

(sec-nct-clock-reflection)=
## 4. A negative reflection matrix inside the native gauge/geometry history

:::{prf:theorem} Actual clock-reflection failure and uniform native transfer discrepancy
:label: thm-nct-clock-reflection-defect

Use the existing reset relation

$$
\lambda=\frac1{a^2(1+c)},\qquad p=0,\qquad
E=\frac c{1+c},\quad D=c-1,
\tag{NCT.8}
$$

and keep any $0<\nu\le\nu_{\rm L}$, all original $q,s,\rho>0$ and
all declared color/observation parameters. For any unit physical component
vector $\ell$, the actual gauge-invariant terminal-position observable
$X_n=\sqrt N\,\ell\cdot\overline x_n$ has stationary covariances

$$
C_{11}=\frac{a^2q^2}{1-c}
       +\frac{4-3c+c^2}{4(1-c)}s^2,\qquad
C_{12}=-\frac{cs^2}{4a(1+c)},
$$
$$
C_{22}=\frac{cq^2}{(1+c)^2(1-c)}
       +\frac{cs^2}{4a^2(1+c)(1-c)},
$$
$$
\Gamma_1=\mathbb E[X_0X_1]=-cs^2/4,
\qquad
\Gamma_2=\mathbb E[X_{-1}X_1]
=-\frac c{1-c}\left[a^2q^2+\frac{3-c}4s^2\right]<0.
\tag{NCT.9}
$$

The complete native common-$SU(3)$ color/geometry trajectory algebra
therefore fails positivity for the sampling-clock reflection
$n\mapsto-n$: its actual one-mode reflected matrix is $[\Gamma_2]$.
A bounded geometry cylinder also has a strictly negative form. This
conclusion uses an observable already in the stated native gauge-neutral
geometry dictionary, rather than the separate ray instrument or a chosen
CAR mode. Its positive-viscosity interacting field law is the full one
constructed above.

Let $C_N$ be the actual centered core transfer and let
$g_N=X_0/\sqrt{C_{11}}$. Its one-mode native injection is an isometry.
For any self-adjoint physical transfer $B_0$ and any verified isometric
native identification $J$ containing this unit mode, $Je=g_N$, one has

$$
\|C_N^2J-JB_0^2\|
\ge -\frac{\Gamma_2}{C_{11}}>0.
\tag{NCT.10}
$$

This includes every proposed positive $B_0=e^{-a_{\rm phys}H}$.
The lower bound is independent of $N$ and of the positive count viscosity
within its proved interval. It is a discrepancy for that actual clock-
reflected color/geometry identification, not a physical mass-gap theorem.
The physical position factor $x_*^2$ multiplies (NCT.9) and cancels from
(NCT.10); the clock calibration fixes the tested separation
$2a_{\rm phys}$. The capped absorbing reference, active-fitness regimes,
color-only trajectory subalgebras, and the separate four-position physical
cut are not inferred to have this same defect.
:::

:::{prf:proof}
At (NCT.8), $A=[\begin{smallmatrix}0&b\\-c/b&c-1\end{smallmatrix}]$.
Insert the three covariance entries in (NCT.9) into
$C=ACA^{\mathsf T}+\Sigma$: each of its three scalar equations holds,
and stability gives uniqueness of this solution. The first lag is
$bC_{12}=-cs^2/4$. The first row of $A^2$ is
$(-c,b(c-1))$, so
$\Gamma_2=-cC_{11}+b(c-1)C_{12}$, giving its displayed strictly
negative value. The field is centered because the stationary centroid is
its zero-mean Gaussian innovation series.

On the two-sided stationary native history take $F=X_1$.
Its reflected quadratic form is exactly
$\mathbb E[\overline{F\circ\vartheta}\,F]
=\mathbb E[X_{-1}X_1]=\Gamma_2$, where
$\vartheta n=-n$. The moment bound supplies $F\in L^2$.
Clipping this original scalar test at $[-R,R]$ converges in $L^2$;
stationarity and Cauchy--Schwarz make its reflected form converge to the
same negative number. Thus a sufficiently large finite $R$ gives a
bounded actual cylinder with negative form. No reflection symmetry of the
whole native law was assumed to obtain this necessary positivity test.

For the claimed operator correspondence,
$\langle g_N,C_N^2g_N\rangle=\Gamma_2/C_{11}<0$, whereas
$\langle e,B_0^2e\rangle=\|B_0e\|^2\ge0$ for a self-adjoint transfer.
Take that unit matrix element of $C_N^2J-JB_0^2$ to obtain (NCT.10).
Only the explicitly normalized native mode is needed for this necessary
comparison; no other unspecified physical injection is assumed to exist.
All covariance identities are independent of the nonlinear relative
count dynamics because the exact native sector correspondence was proved
before the calculation.
:::

(sec-nct-witness)=
## 5. An evaluated genuinely interacting witness and its scope

:::{prf:corollary} Positive-viscosity witness with a nonzero native color field
:label: cor-nct-positive-viscosity-witness

For $d=3$, every permitted $N\ge4$, use the existing unbounded BAOAB
count configuration with

$$
h=\gamma=b_O=\sigma_x=\rho=1,\qquad
\lambda=4/(1+e^{-1}),\qquad \nu=.01,
$$

`velocity_cap=None`, $p_r=p_s=0$, any original positive measurement,
fitness and gate floors, any donor/feature/collision/jitter settings
allowed in the above passive register, any finite $\delta_c\ge0$ and
any $\kappa$. The exact matrix formulas give the diagnostics

$$
H\simeq\begin{pmatrix}1.6244775573&.5365142195\\
.5365142195&2.1584473132\end{pmatrix},\quad
\nu_{\rm L}\simeq.0884804901>\nu,
$$
$$
C_{11}\simeq1.3700034233,\quad
\Gamma_2\simeq-.4458600543,\quad
-\Gamma_2/C_{11}\simeq.3254444819.
\tag{NCT.11}
$$

The exact primitive expressions, rather than rounded decimals, define the
witness. Its full native relative field is nontrivial: the actual B2
color determinant has positive probability of nonzero magnitude for each
finite threshold. This positive finite color is not identified with a
nonzero continuum Yang--Mills field. The negative clock-reflection test
belongs to the existing gauge/geometry history algebra of this interacting
witness. A different physical quotient or four-coordinate reflection
requires its own actual construction; no arbitrary projection or freely
chosen Hamiltonian has replaced this law.
:::

:::{prf:proof}
The values satisfy $0<a^2\lambda<1$. For an exact bound on the viscosity
interval, $1/3<c<3/8$ gives $2/3<b<11/16$. Solving (NCT.2) at the reset
gives

$$
H_{22}=\frac{(1+b^2)(1+c)}{4c(1-c)},\qquad
H_{12}=\frac{1+b^2}{4b},\qquad
H_{11}=1+\frac{c^2}{b^2}H_{22}.
$$

These inequalities give $H_{22}<37323/16384$,
$H_{12}<13/24$ and both absolute row sums below three. Thus $h_+<3$,
while $h_-\ge1$. Direct substitution in $K_0$ gives $K_0<1$.
Consequently the third bound in (NCT.3) exceeds
$(1-\sqrt{2/3})/(2\sqrt3)>1/20$; its other bounds exceed $1/4$.
Hence $\nu=.01<\nu_{\rm L}$ without reliance on diagnostic rounding.
All covariance and Lyapunov values follow by substitution in their small
primitive linear systems. The full-support invariant law and original
first kick supply positive probability of every finite open preparation:
when $a\nu<1$, the map $(x,v)\mapsto(p,v_1)$ is a diffeomorphism.
Indeed its inverse is
$x=p-av_1$ and
$v=(I-a\nu L_x)^{-1}(v_1+a\lambda x)$, with a positive definite
inverse matrix. Hence no target preparation is assumed to occur under an
unknown law.

For a concrete nonzero-color preparation choose B2 positions all zero
and O-stage velocities
$z_1=Le_1$, $z_2=Le_2$, $z_3=Le_3$,
$z_4=-L(e_1+e_2+e_3)$, with remaining $z_i=0$.
Set $p_i=-az_i$ and choose any finite $v_1$, for example zero.
The count force is $F_i=-\nu z_i$, because its kernels are all one and
its velocity mean is zero. Take $L>\delta_c/\nu$.
The first three normalized colors are nonzero multiples of the three
coordinate unit vectors, with phase $e^{i\kappa L}$, so their determinant
has unit magnitude. Continuity gives an open preparation/innovation event
with force norms still above the original threshold and determinant bounded
away from zero. Full preparation support and the original Gaussian O draws
give it positive probability under the actual stationary attached law.
This proves genuine finite nonzero color without changing its threshold,
force or noise. The independent final position innovation and unbounded
boundary do not delete that event.
:::

(sec-nct-history-classification)=
## 6. Exact parameter and recording-stride classification of centroid history

:::{prf:theorem} Solved native centroid-position reflection regimes
:label: thm-nct-centroid-history-classification

Retain the full invariant-law regime of
{prf:ref}`thm-nct-full-native-invariant`, with all its original parameters.
Put $L=a^2\lambda\in(0,1)$, $Q=a^2q^2>0$, $S=s^2>0$ and
$T=(1+c)(1-2L)$. The stationary one-component position covariance is
exactly

$$
\begin{gathered}
U=C_{11}=\frac{4Q+S[(1-c)^2+L(3-c)(1+c)]}{4L(1-c^2)},\\
aC_{12}=-(1-L)S/4,\qquad
C_{22}=\frac{(1-L)[4Q+LS(1+c)^2]}{4a^2(1-c^2)},\\
\Gamma_0=U,\quad
\Gamma_1=[1-L(1+c)]U-(1+c)(1-L)S/4,\\
\Gamma_{n+2}=T\Gamma_{n+1}-c\Gamma_n.
\end{gathered}
\tag{NCT.12}
$$

For any actual existing measurement stride $m\ge1$, keep the process at
its recorded times $0,m,2m,\ldots$ and its physical step
$ma_{\rm phys}$. This may use the existing `MeasurementConfig.stride`
convention in `physics/spectroscopy/config.rs` when its incoming record
contains consecutive native updates. A pre-strided input record retains
its actual native step gaps and calibration. The integer $m$ here is the
actual gap in complete native updates, not merely an index in an already
thinned array. It does not change any intermediate native update. Define

$$
b_0=0,\quad b_1=1,\quad b_{j+1}=Tb_j-cb_{j-1},
\qquad
\mathscr D=\Gamma_0\Gamma_2-\Gamma_1^2.
$$

The actual sampled one-mode history Hankel determinant is

$$
\Gamma_0\Gamma_{2m}-\Gamma_m^2=b_m^2\mathscr D,
\tag{NCT.13}
$$

where its explicit primitive value is strictly negative:

$$
\begin{aligned}
\mathscr D=-\frac{1-L}{4L(1-c)^2}\bigl\{
 &4Q^2+[(1-c)^2+2L(3-c)(1+c)]QS\\
 &+L[(1-c)^2(1+c)+L(2-c)(1+c)^2]S^2\bigr\}<0.
\end{aligned}
\tag{NCT.14}
$$

Consequently every stride with $b_m\ne0$ fails sampling-clock reflection
positivity within the actual centroid-position history algebra. The test
can use strictly positive recorded times $m,2m$: its reflected two-by-two
matrix has determinant $c^{2m}b_m^2\mathscr D<0$.

All remaining cases are explicitly as follows. Real distinct or repeated
roots of $z^2-Tz+c$ have $b_m\ne0$ at every positive stride, and hence
never give a positive centroid-history reconstruction in this regime.
For complex roots $\sqrt c\,e^{\pm i\theta}$, $0<\theta<\pi$,

$$
\cos\theta=\frac{(1+c)(1-2L)}{2\sqrt c},\qquad
b_m=c^{(m-1)/2}\frac{\sin(m\theta)}{\sin\theta}.
\tag{NCT.15}
$$

At the exact resonances $m\theta=k\pi$, the actual native centroid
transfer at that stride is the scalar autoregression
$A^m=r_mI$, $r_m=(-1)^kc^{m/2}$.
Odd $k$ gives site reflection positivity but fails link reflection
positivity and the positive-energy transfer identification.
Even $k$ gives a genuine positive native centroid-history transfer and
physical Hamiltonian as proved below. Thus its solved positive-energy
regime is exactly

$$
\theta=2\pi j/m\in(0,\pi),\qquad
L=\frac12\left[1-\frac{2\sqrt c}{1+c}\cos(2\pi j/m)\right].
\tag{NCT.16}
$$

Every integer $m\ge3$ and $1\le j<m/2$ gives an explicitly evaluated
existing parameter family, subject to the already proved positive
viscosity interval. Recording every intermediate step retains the earlier
negative test; positivity belongs to the declared strided observation
algebra. No positivity of the full native color-history algebra is inferred
from this gauge-neutral subalgebra.
:::

:::{prf:proof}
Insert the covariance entries in (NCT.12) into the three equations
$C=ACA^{\mathsf T}+\Sigma$. The entry $C_{12}$ is the negative
position-noise contribution; the original O and force terms cancel exactly.
The equation for the covariance sequence follows from the characteristic
identity $A^2-TA+cI=0$. Substitution of $\Gamma_0,\Gamma_1$ in
$\mathscr D=U(T\Gamma_1-cU)-\Gamma_1^2$ gives (NCT.14).
Every bracket coefficient is positive for $0<c,L<1$, and $Q>0$.
This proves strict negativity without an unknown stationary-law estimate.

For distinct roots $\lambda_\pm$, solving the actual covariance recurrence
gives $\Gamma_n=w_+\lambda_+^n+w_-\lambda_-^n$, with

$$
w_+=\frac{\Gamma_1-\lambda_-\Gamma_0}{\lambda_+-\lambda_-},
\qquad
w_-=\frac{\lambda_+\Gamma_0-\Gamma_1}{\lambda_+-\lambda_-}.
$$

Its stride-$m$ determinant is
$w_+w_-(\lambda_+^m-\lambda_-^m)^2$, while
$\mathscr D=w_+w_-(\lambda_+-\lambda_-)^2$.
The primitive recurrence gives
$b_m=(\lambda_+^m-\lambda_-^m)/(\lambda_+-\lambda_-)$,
proving (NCT.13); a repeated-root limit gives
$b_m=m\lambda^{m-1}\ne0$ because $\lambda^2=c>0$.
Distinct real roots have the same sign and distinct absolute values, so
no power equality is possible. The complex-root formula gives (NCT.15).

Cayley--Hamilton also gives
$A^m=b_mA-cb_{m-1}I$. Hence $b_m=0$ makes $A^m=r_mI$ with the
stated sign and magnitude. If $b_m\ne0$, the reflected covariance matrix
at times $m,2m$ is
$[\begin{smallmatrix}\Gamma_{2m}&\Gamma_{3m}\\
\Gamma_{3m}&\Gamma_{4m}\end{smallmatrix}]$.
Its determinant is $(\lambda_+^m\lambda_-^m)^2$ times the stride-zero
one, namely $c^{2m}b_m^2\mathscr D<0$, with the same repeated-root
limit. A negative eigenvector supplies a native finite linear history test;
its bounded clipping has a negative form by the finite Gaussian moments.
This uses actual strictly future cylinders.

At a resonance the Gaussian chain at the measured stride has covariance
$C$ and innovation covariance $(1-r_m^2)C$. Its position coordinate is
therefore a closed Gaussian autoregression with coefficient $r_m$.
It is reversible. Conditional past/future independence gives site
reflection positivity. For $r_m<0$, the one-step link-reflected linear
form is $\Gamma_m=r_mU<0$, so link positivity and a positive transfer
fail. For $r_m>0$, both positivities and positive-energy reconstruction
follow from the actual Mehler operator in the next theorem. Solving its
complex angle relation for $L$ gives (NCT.16). Since
$2\sqrt c/(1+c)<1$, every such $L$ lies strictly between zero and one.
The full interacting invariant-law budget remains nonempty there.
:::

(sec-nct-positive-physical-centroid)=
## 7. Actual positive-energy reconstruction on the strided centroid algebra

:::{prf:theorem} Native Mehler correspondence and a population-uniform physical centroid gap
:label: thm-nct-positive-centroid-reconstruction

In the exact positive resonance (NCT.16), write $r_m=c^{m/2}\in(0,1)$
and normalize the actual recorded position field by
$\mathcal X=\sqrt N\,\overline x/\sqrt U$.
Its stationary law is standard $d$-Gaussian, and its actual transfer at
recording stride $m$ is

$$
(K_{r_m}f)(z)=\mathbb E f(r_m z+\sqrt{1-r_m^2}Z_d).
\tag{NCT.17}
$$

The Gaussian position-history reflection quotient is isometric to
$L^2(N(0,I_d))$. Under that isometry its actual time translation is
$K_{r_m}$. It is self-adjoint and positive, with normalized Hermite
multi-indices $\alpha$ as a complete eigenbasis and eigenvalues
$r_m^{|\alpha|}$. Its unique vacuum is the constant function. The
centered spectrum consists of those positive point eigenvalues for
$|\alpha|\ge1$ and the accumulation point zero; zero has no eigenvector.

The same-law physical Hamiltonian is thus defined, rather than assumed,
by its actual transfer correspondence:

$$
H_{\rm cent}=-\frac{\hbar_{\rm eff}}{ma_{\rm phys}}\log K_{r_m},
\qquad
H_{\rm cent}\mathsf H_\alpha
=\frac{\hbar_{\rm eff}\gamma}{2t_*}|\alpha|\mathsf H_\alpha,\qquad
K_{r_m}=e^{-ma_{\rm phys}H_{\rm cent}/\hbar_{\rm eff}}.
\tag{NCT.18}
$$

Its physical gap is exactly
$\hbar_{\rm eff}\gamma/(2t_*)>0$, independent of $N$ and of the
interacting positive viscosity in its proved interval. The native
position injection $Jf=f(\sqrt N\overline x/\sqrt U)$ is explicitly
isometric and satisfies $P_N^mJ=JK_{r_m}$; its range is a reducing
subspace of this actual core transfer. This is a positive physical correspondence on the stated existing
centroid-position observation algebra. It is not the physical local
Yang--Mills sector or its target mass-gap identification. Native colors,
intermediate updates, alive/dead variants and other physical reflections
retain their separate tests.
:::

:::{prf:proof}
The actual centroid chain gives
$Z_{(n+1)m}=r_mZ_{nm}+\eta_n^{(m)}$, with independent innovation
covariance $(1-r_m^2)C$. Its position marginal is therefore exactly the
chain (NCT.17), independent of past position history. Composition with
this actual Gaussian pushforward law makes $J$ isometric, and the
conditional future mean/law gives the asserted intertwining.
For the reducing assertion, split the stationary centroid velocity into
its conditional linear mean $(C_{12}/C_{11})\overline x$ and its Gaussian
residual. That residual is independent of the position coordinate. At the
resonance both have the same autoregression coefficient $r_m$, and their
innovation blocks are independent because their covariance is proportional
to the same stationary $C$. The already proved relative-coordinate factor
is independent as well. Conditional expectation onto centroid position
therefore commutes with the actual $m$-step core transfer and its adjoint.
This is a derived native invariant sector, rather than an imposed gauge
projection. It is not a reducing assertion for an augmented space that
also retains arbitrarily informative past passive records.

For a bounded future position cylinder $F$ starting at sampled time zero,
put $V_F(z)=\mathbb E[F\mid\mathcal X_0=z]$ under this actual stationary
chain. The Mehler joint Gaussian of consecutive observations is symmetric.
Its reversed transition is its forward transition. Conditional independence
of its past and future then gives the site-reflected inner product
$\mathbb E[\overline{F\circ\vartheta}G]
=\langle V_F,V_G\rangle_{L^2(N(0,I_d))}$.
Every bounded function of time zero occurs as such a $V_F$ and is dense in
that Hilbert space. Quotienting the null vectors therefore gives precisely
this physical history space. Shifting the same future cylinder by one
recorded step sends $V_F$ to $K_{r_m}V_F$, proving the transfer identity.
There is no assigned physical exponential in this construction.

The Gaussian generating identity
$\exp(t\cdot z-|t|^2/2)=\sum_\alpha t^\alpha
\mathsf{He}_\alpha(z)/\alpha!$ gives, by applying (NCT.17), the same
identity with $t$ replaced by $r_mt$. Thus normalized Hermite polynomials
are eigenvectors with eigenvalues $r_m^{|\alpha|}$. Their orthogonality
follows by multiplying the generating identities and Gaussian integration.
They are complete: if a Gaussian-$L^2$ function is orthogonal to every
polynomial, its Gaussian-weighted exponential transform is analytic by
Cauchy--Schwarz and has all derivatives zero at zero. Its Fourier transform
therefore vanishes; Fourier uniqueness makes the original function zero.
Hence this actual transfer is positive and self-adjoint. Each degree has
finite multiplicity, so its eigenvalues tend to zero and no eigenvalue is
zero. For a link cut, condition the future cylinder on its first future sampled
time; its conditional function has the same role as $V_F$ above.
The link-reflected form is
$\langle V_F,K_{r_m}V_F\rangle\ge0$, proving the other positivity needed
for the stated positive transfer.

The logarithm is defined on the Hermite eigenbasis with domain consisting
of square-summable coefficients weighted by $|\alpha|^2$. It is
self-adjoint and nonnegative. Since
$-\log r_m=(m/2)\gamma h$ and $a_{\rm phys}=t_*h$, (NCT.18) and its
exact gap follow. All normalizations consume actual native coordinates,
population and calibration fields; no alteration of the interacting
relative dynamics or new Gaussian trajectory supplies this sector.
:::

:::{prf:corollary} Evaluated positive stride-three witness
:label: cor-nct-positive-stride-three

Choose the same existing unbounded count/constant-fitness/cap-none
configuration, with $d=3$, $N\ge4$, $h=b_O=\sigma_x=\rho=1$,

$$
c=(3-\sqrt5)/2,\quad\gamma=-\log c,\quad
\lambda=4/(1+c),\quad\nu=.01,
\quad\text{actual measurement stride }m=3.
\tag{NCT.19}
$$

Retain all original remaining configuration, color threshold, phase,
arithmetic and observation fields as above. Here
$\sqrt c=(\sqrt5-1)/2$, $p=0$ and
$\operatorname{tr}A=c-1=-\sqrt c=2\sqrt c\cos(2\pi/3)$.
Thus $A^3=c^{3/2}I$ exactly and the actual recorded centroid-position
history has the positive transfer (NCT.17). Its normalized physical gap is
$\hbar_{\rm eff}[-\log((3-\sqrt5)/2)]/(2t_*)$.
The full interacting invariant law and the finite nonzero-color event
proved above apply; the color-only and full color-history positivity
questions are not discharged by this centroid reconstruction.
:::

:::{prf:proof}
The chosen $c$ lies in $(1/3,2/5)$, and its exact reset makes
$0<L<1$. The explicit (NCT.2) matrix has $h_+<3$, $h_-\ge1$ and
$K_0<1$ by the same reset formulas; substituting
$c=(3-\sqrt5)/2$ verifies these inequalities directly.
Consequently $\nu_{\rm L}>(1-\sqrt{2/3})/(2\sqrt3)>1/20$,
so the chosen positive viscosity is admitted. Its angle is exactly
$2\pi/3$, which proves the scalar third-step map and the positive
Hamiltonian correspondence. Every intermediate step remains the original
native update. The primitive measurement stride selects the declared
observable algebra on which that correspondence holds.
:::


:::{prf:remark} Actual initial laws and retained intermediate records
:label: rem-nct-initial-and-observation-scope

The stationary formulas use the unique native invariant law constructed
above. For the actual initial centroid $Z_0$ in the complete execution
record, the exact transient remains

$$
Z_n=A^nZ_0+\sum_{j=0}^{n-1}A^{n-1-j}\eta_j.
$$

Stability makes its first term converge to zero almost surely for every
finite-coordinate initial state, and its Gaussian innovation term converges
to the stationary centroid law. If that initial law has finite second
moments, its actual covariance is
$A^nC_0(A^n)^{\mathsf T}+\sum_{j<n}A^j\Sigma(A^j)^{\mathsf T}$.
Thus a finite recorded window is not silently assigned stationary
covariances or a finite burn-in certificate. The strided positive sector
uses its actual declared observation schedule. The same execution with
intermediate times retained has the proved negative centroid-history
reflection test. Neither observation choice changes the gas transition.
:::
