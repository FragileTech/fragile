# Native interaction transport, conditional action, and a proved face-action limit

(sec-nyc-ledger)=
## 1. Complete execution and the existing interaction readout

:::{prf:definition} Complete record for the native connection calculation
:label: def-nyc-complete-record

Retain the complete execution record $\mathfrak P$ in
{prf:ref}`def-native-complete-execution-record`, the restricted reference
ledger in {prf:ref}`def-native-jg-ledger`, and the source-stage convention in
{prf:ref}`def-native-ym-execution-ledger`. All initial, donor, gate, reward,
collision, landscape, noise, boundary, recording, geometry, arithmetic,
allocation and calibration fields remain in these records. In particular,
every preceding simultaneous copying decision and collision-component rotation
is part of the preparation below. The real-coordinate statements use the
existing independent Gaussian innovation convention, not a fixed seed.

The first calculation uses the existing dense **count-normalized** Gaussian
viscosity, quadratic potential $U(x)=\lambda|x|^2/2$, terminal classification,
isotropic OU noise, independent final position diffusion, and the configured
radial velocity cap

$$
C_V(w)=\frac{Vw}{V+|w|},\qquad V>0.
$$

This is the cap implemented in `kinetic.rs`; it is not a hard truncation.
No graph-force, curl, intermediate absorption or geometry-feedback stage is
enabled in this restriction. Subsequent face-limit calculations also cover
the existing row normalization and the configured smooth polynomial potential
providers, with their actual primitive values retained.

The connection readout is the existing executed-record readout in
`physics/qft/run_observables.rs`, experiments $1,11,15,23,24,29$:

$$
r(x,v)=\frac{x+iv}{\sqrt{|x|^2+|v|^2}},\qquad
\ell_{ab}=\frac{r_a^\dagger r_b}{|r_a^\dagger r_b|}.
$$

It uses actual pre-update and final-update events, their slot, generation and
version labels, the configured eligible mask, the ray-norm threshold
$10^{-12}$, and overlap threshold $10^{-10}$. A triangle is used only when
all three events and overlaps are available. Its oriented holonomy and
measured face defect are

$$
H_{abc}=\ell_{ab}\ell_{bc}\ell_{ca},\qquad w_{abc}=1-\Re H_{abc}.
$$

The code adds the stored numerical position and velocity coordinates in this
formula. A change of their relative scale changes this readout and must be
recorded; the physical speed and action calibration does not silently alter
it. The algebraic field descriptor may retain every such individual face,
not only its final normalized average.
For a fixed tagged face, an unavailable phase is extended by zero together
with its retained availability indicator. This is the pullback of that
tagged component of the returned available-face list; it does not replace
the list's actual empty-average convention.

For one step let $\mathcal H$ be its complete actual preparation through A1.
Write its A1 positions and pre-O velocities as $p,v_1$, and put

$$
a=h/2,\quad c=e^{-\gamma h},\quad
q^2=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,
\end{cases}\quad
s^2=\sigma_x^2h,\quad m=p+acv_1,\quad \tau^2=a^2q^2+s^2.
$$

The B2 force inputs and final positions are exactly

$$
z=cv_1+q\xi,\qquad X=p+az=m+aq\xi,
\qquad Y=m+aq\xi+s\zeta,
$$

where $\xi,\zeta$ are the actual independent standard normal arrays. For
the explicit conditional density assume the primitive tests $q>0$ and
$s>0$. Write

$$
\chi=\frac{s^2}{\tau^2},\qquad
\bar z=cv_1+q\frac{aq}{\tau^2}(y-m),\qquad k=Nd.
$$

Thus $z\mid(\mathcal H,Y=y)$ has the already proved normal law
$N(\bar z,\chi q^2I_k)$. The cap and B2 change no terminal position.
Alive/dead marks and a one-step survival event are fixed at this fiber.
The actual geometry-only posterior correction on a coarser descriptor
remains the term in
{prf:ref}`thm-native-ym-preparation-posterior-response`.
:::

(sec-nyc-b2-inverse)=
## 2. A primitive global inverse for the executed B2 map

:::{prf:theorem} Global B2 inverse under an evaluated prepared-position test
:label: thm-nyc-global-b2-inverse

Use {prf:ref}`def-nyc-complete-record`. For fixed preparation define

$$
\begin{aligned}
\mathcal G_i(z)&=\frac1N\sum_{j\ne i}
 \exp\!\left[-\frac{|p_i-p_j+a(z_i-z_j)|^2}{2\rho^2}\right](z_i-z_j),\\
\mathcal K(z)&=(1-a^2\lambda)z-a\lambda p-a\nu\mathcal G(z),\\
D_p&=\max_{i,j}|p_i-p_j|,\qquad
C_p=1+\frac2e+\frac{D_p}{\rho\sqrt e},\\
\beta_-&=1-a^2\lambda-2a\nu C_p,\qquad
\beta_+=1-a^2\lambda+2a\nu C_p.
\end{aligned}
$$

Here $h>0$, $\rho>0$, $\nu\ge0$, $\lambda\ge0$ are the original parameters.
If the derived test $\beta_->0$ holds, the actual uncapped B2 map
$w=\mathcal K(z)$ is a global $C^\infty$ bijection of $\mathbb R^k$.
Its Jacobian $J(z)=D\mathcal K(z)$ has positive determinant and satisfies

$$
\beta_-|z-z'|\le|\mathcal K(z)-\mathcal K(z')|
 \le\beta_+|z-z'|,
\qquad
\beta_-\le\sigma_j(J(z))\le\beta_+.
$$

In particular its inverse is determined by a convergent fixed-point iteration
using the native map,

$$
z_{n+1}=\frac{w+a\lambda p+a\nu\mathcal G(z_n)}{1-a^2\lambda},
$$

with contraction factor at most
$2a\nu C_p/(1-a^2\lambda)<1$.
There is no bound on an OU draw in this result.
:::

:::{prf:proof}
The actual second half-kick is
$w=z+a[-\lambda(p+az)-\nu\mathcal G(z)]$, giving $\mathcal K$.
For $d_{ij}=z_i-z_j$, $b_{ij}=p_i-p_j$ and
$R_{ij}=b_{ij}+ad_{ij}$, differentiation of the corresponding pair term
gives the matrix

$$
M_{ij}=e^{-|R_{ij}|^2/(2\rho^2)}
 \left[I-\frac a{\rho^2}d_{ij}R_{ij}^{\mathsf T}\right].
$$

Since $ad_{ij}=R_{ij}-b_{ij}$, its norm is at most

$$
e^{-|R|^2/(2\rho^2)}
 +\frac{|R|^2+D_p|R|}{\rho^2}e^{-|R|^2/(2\rho^2)}
\le1+\frac2e+\frac{D_p}{\rho\sqrt e}=C_p.
$$

The last inequality follows by maximizing $r^2e^{-r^2/(2\rho^2)}$
and $re^{-r^2/(2\rho^2)}$ at $r=\sqrt2\rho$ and $r=\rho$.
The block Jacobian of $\mathcal G$ has off-diagonal blocks $-M_{ij}/N$
and diagonal blocks $N^{-1}\sum_{j\ne i}M_{ij}$. Moreover
$M_{ji}=M_{ij}$, so both its row sums and column sums of block norms are
at most $2C_p$. For any block vector, scalar Cauchy--Schwarz with these
nonnegative block-norm coefficients proves
$\|D\mathcal G\|_2\le2C_p$. This argument does not assume that $M_{ij}$
is symmetric as a matrix.

Integrating the Jacobian along a line segment proves
$|\mathcal G(z)-\mathcal G(z')|\le2C_p|z-z'|$. The reverse triangle
inequality applied to $\mathcal K$ gives its lower bound; the ordinary
triangle inequality gives its upper bound. Set
$\alpha=1-a^2\lambda>0$. For each $w$ the displayed fixed-point map
has contraction factor $2a\nu C_p/\alpha<1$. The successive differences
of its iterates are bounded by a geometric series, so the iterates are
Cauchy in $\mathbb R^k$. Continuity gives a fixed point, and the same
contraction bound gives uniqueness. This proves surjectivity and injectivity.
The bounds on singular values follow from
$J=\alpha I-a\nu D\mathcal G$. The continuous matrices
$\alpha I-t a\nu D\mathcal G(z)$, $0\le t\le1$, are nonsingular,
and their determinant begins at $\alpha^k>0$. Thus $\det J>0$.
The nonsingular derivative and the ordinary local inverse theorem give
a smooth inverse near every point; uniqueness makes these inverses agree
globally. All estimates hold for every real $z$.
:::

:::{prf:corollary} The unchanged reference passes the global inverse test
:label: cor-nyc-reference-inverse

In {prf:ref}`def-cgd-existing-reference`, $D=[-2,2]^3$, $h=.04$,
$\lambda=1$, $\nu=.3$, $\rho=1$, $V=2$, $\alpha_{\rm col}=.5$,
and $\sigma_J=.1$. The actual pre-collision cap and collision yield
$|v_i^J|\le R_c=(1+2|\alpha_{\rm col}|)V=4$.
Let $J_0=.8$ be a proof radius, and let $\mathcal E_{J_0}$ mean that
every accepted or forced clone jitter in the present preparation has length
at most $J_0$. Conditional on every actual entering nonextinct state,

$$
P(\mathcal E_{J_0}^{\rm c}\mid\text{entering state})
\le N\,2^{3/2}e^{-J_0^2/(4\sigma_J^2)}.
$$

On this event the actual A1 preparation satisfies

$$
\begin{aligned}
X_J&=2\sqrt3+.8=4.2641016151\ldots,\\
V_1&=4+2(.02)(.3)4+.02X_J=4.1332820323\ldots,\\
X_1&=X_J+.02V_1=4.3467672558\ldots,\qquad D_p\le2X_1,\\
C_p&\le7.0086541049\ldots,\qquad
\beta_-\ge .9154961507\ldots,\quad
\beta_+\le1.0837038493\ldots.
\end{aligned}
$$

For $N=200$ the failed-event bound is $6.365950814\times10^{-5}$.
This is a quantitative probability of an invertible **whole B2 map**;
it is not a clipped-noise algorithm. For any fixed finite preparation,
the same test holds for all sufficiently small $h>0$. For the existing row
normalization the density result below is not inferred from this count test;
failure of the count test outside this event is not a proof of noninvertibility.
:::

:::{prf:proof}
Every donor position is in the actual eligible domain $D$ before simultaneous
replacement. A persistent position remains in $D$, and a replaced one is
within its actual jitter of a donor in $D$. Thus $|x_i^J|\le X_J$.
The original collision bound gives $R_c$ without altering a sampled rotation.
The first viscous force has magnitude at most $2\nu R_c$, because the count
sum of the Gaussian weights is at most one. The first potential force is
at most $\lambda X_J$. The first half-kick therefore gives the displayed
$V_1$, and A1 gives $X_1$. Substitute $D_p\le2X_1$ in the preceding
theorem. For a Gaussian jitter $\eta\sim N(0,\sigma_J^2 I_3)$,

$$
P(|\eta|>J_0)\le e^{-J_0^2/(4\sigma_J^2)}
 E e^{|\eta|^2/(4\sigma_J^2)}
=2^{3/2}e^{-J_0^2/(4\sigma_J^2)}.
$$

There are at most $N$ actual replaced rows. Conditional on donor and gate
choices, apply the union bound to their existing draws, and then average.
No independence of donor choices is used. For a fixed preparation its finite
$D_p$ makes $2a\nu C_p+a^2\lambda\to0$ as $h\downarrow0$.
:::

(sec-nyc-explicit-native-action)=
## 3. The actual capped conditional likelihood and its first variation

:::{prf:theorem} A uniform terminal-target inverse across all unbounded clone jitters
:label: thm-nyc-uniform-target-inverse

Keep the count-quadratic restriction of
{prf:ref}`def-nyc-complete-record`, with arbitrary finite actual A1
preparation $p,v_1$. For a target radius $W\ge0$ define the primitive
constants

$$
E_W=\frac{W+a^2\lambda\nu\rho/\sqrt e}{1-2a\nu},\qquad
C_W=1+\frac{2a^2\lambda}e+\frac{2aE_W}{\rho\sqrt e},\qquad
b_W=1-a^2\lambda-2a\nu C_W.
$$

If $2a\nu<1$ and $b_W>0$, then for **every** preparation and every target
$\max_i|w_i|\le W$ the actual B2 equation $\mathcal K(z)=w$ has exactly
one real solution. At every such solution

$$
\sigma_{\min}(D\mathcal K)\ge b_W,\qquad
\sigma_{\max}(D\mathcal K)\le1-a^2\lambda+2a\nu C_W,
\qquad \det D\mathcal K>0.
$$

These constants do not contain the prepared positions, donor identities,
gate pattern, collision rotation or Gaussian clone-jitter lengths.
There is no conditioning that bounds those jitters.

For $q>0$, $s>0$, final velocity cap $V>0$ and target $u\in\mathcal B_V$
with $\max_i|Vu_i/(V-|u_i|)|\le W$, the actual joint terminal density
at $(Y=y,u)$, conditional on **any** such complete preparation, is

$$
\begin{aligned}
p_{\mathcal H}(y,u)
={}&\frac{\exp[-|z-cv_1|^2/(2q^2)
                 -|y-p-az|^2/(2s^2)]}
 {(2\pi q s)^k\det D\mathcal K(z)}
 \prod_i\left(\frac V{V-|u_i|}\right)^{d+1},\\
p_{\mathcal H}(y,u)
\le{}&(2\pi q s)^{-k}b_W^{-k}
 \prod_i\left(\frac V{V-|u_i|}\right)^{d+1}.
\end{aligned}
$$

Consequently every actual mixture over accepted-clone patterns, donors,
jitters and collision rotations has the same local upper bound, multiplied
only by that mixture's **actual total probability** when it is a sublaw.
This includes the full, unmarked terminal law and every subset of cloning
patterns. On a compact target $|u_i|\le U<V$, take
$W=VU/(V-U)$ and replace the cap factor by
$(V/(V-U))^{N(d+1)}$. This yields a uniform bound even as the entering
state approaches a consensus configuration.

The unchanged reference gives, for $W=1$ and $U=2/3$,

$$
E_W=1.0122194167\ldots,\quad C_W=1.0248519880\ldots,
\quad b_W=.9873017761\ldots>0.
$$

In particular the uniform target density bound applies to its entire
Gaussian-jitter law at every population size for which the actual full
execution is defined. This result does not assert a global inverse outside
the stated target region.
:::

:::{prf:proof}
At a solution, let $X=p+az$ and let $L_X$ be the existing count
Laplacian, so $\mathcal G(z)=L_Xz$. The native equation can be written

$$
w=(I-a\nu L_X)z-a\lambda X.
$$

Set $B_X=I-a\nu L_X$. In the maximum block norm,
$\|L_X\|_{\infty}\le2$, so its Neumann series gives
$\|B_X^{-1}\|_\infty\le(1-2a\nu)^{-1}$. Moreover

$$
|(L_XX)_i|\le\frac1N\sum_{j\ne i}
 |X_i-X_j|e^{-|X_i-X_j|^2/(2\rho^2)}\le\frac\rho{\sqrt e}.
$$

Thus

$$
z=a\lambda X+e,\qquad
e=B_X^{-1}w+a^2\lambda\nu B_X^{-1}L_XX,
\qquad \max_i|e_i|\le E_W.
$$

For the pair Jacobian in the preceding theorem,
$z_i-z_j=a\lambda(X_i-X_j)+(e_i-e_j)$. Substitution gives

$$
\|M_{ij}\|\le e^{-R^2/(2\rho^2)}
 \left[1+\frac{a^2\lambda R^2+2aE_WR}{\rho^2}\right]
\le C_W.
$$

The same block row/column argument gives
$\|D\mathcal G\|_2\le2C_W$ **at every preimage of this target**,
which proves the singular-value estimates there.

For existence and uniqueness retain the homotopy
$\mathcal K_t(z)=(1-a^2\lambda)z-a\lambda p-ta\nu\mathcal G(z)$,
$0\le t\le1$. The preceding estimates hold at its target preimages
with $\nu$ replaced by $t\nu$ and are bounded by the displayed $E_W,C_W$.
Since $L_X$ is symmetric with spectrum in $[0,1]$,

$$
|\mathcal K_t(z)+a\lambda p|
\ge(1-a^2\lambda-a\nu)|z|.
$$

The coefficient is positive because $b_W>0$ and $C_W\ge1$.
Thus all possible preimages of a fixed $w$, throughout this homotopy,
lie in one bounded set determined by $w,p$. At $t=0$ there is exactly
one preimage. Its invertible Jacobian continues this solution locally in
$t$, and on the continued solution

$$
\frac{dz}{dt}=(D\mathcal K_t)^{-1}a\nu\mathcal G(z).
$$

The inverse derivative is bounded by $b_W^{-1}$ and
$|\mathcal G(z)|=|L_Xz|\le|z|$. Hence this derivative is uniformly
bounded on the bounded preimage set. The solution cannot escape or stop:
at a finite endpoint it has a limit, the defining equation passes to that
limit, and the same nonsingular Jacobian continues it. It reaches $t=1$.
Conversely, any preimage at $t=1$ can be continued backwards by the same
argument to the unique preimage at $t=0$. Local uniqueness for this smooth
continuation equation makes the two paths coincide. This proves exactly
one preimage for every target in question. Its determinant remains
positive along the continued solution, beginning at
$(1-a^2\lambda)^k$. Local inversion also proves smooth dependence on
$(w,p)$ within the strict target region.

Conditional on preparation, the actual independent variables $z,\zeta$
have density
$(2\pi)^{-k}q^{-k}\exp[-|z-cv_1|^2/(2q^2)-|\zeta|^2/2]$.
The change of variables $(z,\zeta)\mapsto(w,Y)$ has block derivative
$\left(\begin{smallmatrix}D\mathcal K&0\\aI&sI\end{smallmatrix}\right)$
and determinant $s^k\det D\mathcal K$. There is exactly one preimage
on the stated target. Composing with the true radial cap inverse gives
the density formula. Drop its exponential, use
$\det D\mathcal K\ge b_W^k$, and then integrate each preceding
conditional mixture. No factor of that mixture is assumed independent,
and every conditional jitter and Haar kernel is integrated with total
mass one. A sublaw contributes its original total mass.
:::

:::{prf:theorem} Explicit geometry-fiber action in the native final-velocity chart
:label: thm-nyc-explicit-capped-action

Use the primitive regime in {prf:ref}`thm-nyc-global-b2-inverse`. Let
$u=(u_1,\ldots,u_N)$ be the **actual final velocity** and set

$$
\mathcal B_V=\{u:|u_i|<V\ \text{for every }i\},\qquad
w_i(u)=\frac{Vu_i}{V-|u_i|},\qquad z(u)=\mathcal K^{-1}(w(u)).
$$

On the fixed actual terminal-position fiber $(\mathcal H,Y=y)$ its full
velocity density relative to $k$-dimensional Lebesgue volume is

$$
p_0^{\mathcal H,y}(u)
=\frac{\exp[-|z(u)-\bar z|^2/(2\chi q^2)]}
 {(2\pi\chi q^2)^{k/2}\det J(z(u))}
 \prod_{i=1}^N\left(\frac V{V-|u_i|}\right)^{d+1},
\qquad u\in\mathcal B_V.
$$

It is positive throughout this open product of balls and integrates to one.
The actual action in this chart is explicitly

$$
S_0^{\mathcal H,y}(u)
=\frac{|z(u)-\bar z|^2}{2\chi q^2}
 +\log\det J(z(u))
 +(d+1)\sum_i\log\left(1-\frac{|u_i|}V\right)
 +\frac k2\log(2\pi\chi q^2).
$$

For the existing addressed O source $\xi\mapsto\xi+\theta f$,
$f\in\mathbb R^k$, its **conditional** density on the same fiber obeys

$$
\frac{p_\theta^{\mathcal H,y}(u)}{p_0^{\mathcal H,y}(u)}
=\exp\left[\frac\theta q f\cdot(z(u)-\bar z)
                  -\frac{\theta^2\chi|f|^2}2\right].
$$

In particular the source score and native coordinate tangent are

$$
M_\perp(u)=\frac1q f\cdot(z(u)-\bar z),\qquad
X_f(u)=\chi q\,DC_V(w(u))J(z(u))f.
$$

For every bounded $C^1$ test $O$ with bounded derivative on
$\mathcal B_V$,

$$
\int X_f\cdot\nabla O\,p_0^{\mathcal H,y}\,du
=\int O M_\perp\,p_0^{\mathcal H,y}\,du.
$$

Away from the zero-row coordinate sets, the pointwise action identity is

$$
X_f\cdot\nabla S_0^{\mathcal H,y}
 -\operatorname{div}_{du}X_f=M_\perp.
$$

Thus the likelihood, cap Jacobian, B2 Jacobian and reference-volume
divergence are identified in the **actual final-velocity coordinates**.
The score for any retained ray, interaction-loop or gauge descriptor $D$
is the pushforward $E[M_\perp\mid\mathcal H,y,D]$ under this explicit
positive density. The displayed $S_0$ is the fine conditional action;
integrating this density over an existing descriptor is still required to
obtain its coarser action.
:::

:::{prf:proof}
The configured cap is a radial bijection $\mathbb R^d\to B_V$.
At $r=|w|>0$ its tangential derivative has eigenvalue $V/(V+r)$,
and its radial derivative has eigenvalue $V^2/(V+r)^2$. Its determinant
is therefore $(V/(V+r))^{d+1}$. The derivative is $I$ at zero, and
the same determinant formula holds there by continuity. Inverting
$|u|=Vr/(V+r)$ gives the displayed $w(u)$ and

$$
\det D_uw_i(u)=\left(\frac V{V-|u_i|}\right)^{d+1}.
$$

Apply change of variables first through the global native B2 inverse and
then through this actual cap, to the conditional density
$N(\bar z,\chi q^2 I_k)$. This gives $p_0$ and its normalization, and
taking its negative logarithm gives $S_0$. No cap boundary atom occurs:
each finite real innovation maps into the open velocity ball.

At fixed $(\mathcal H,y)$ the source changes $\bar z$ by
$\theta\chi q f$, with unchanged covariance and unchanged map
$\mathcal K$. Dividing the two Gaussian densities proves the ratio.
Its score is $M_\perp$. Translate $z$ by $\theta\chi q f$ before applying
the same $C_V\mathcal K$. Differentiating this map gives $X_f$.
Its derivative is bounded by $\chi q\beta_+|f|$, so the translated test
has an integrable derivative. Gaussian integration by parts, or
differentiation of the exact density ratio, proves the weak identity.

To verify the pointwise identity, put $T=C_V\mathcal K$ and
$A(z)=\det DT(z)$. On each smooth chart with nonzero cap-input rows,
$S_0(T(z))=|z-\bar z|^2/(2\chi q^2)+\log A(z)
+k\log(2\pi\chi q^2)/2$. The coordinate tangent is $DT\,\chi q f$.
The change-of-volume formula for the divergence of this pushed constant
vector is
$\operatorname{div}_{du}X_f=\chi q f\cdot\nabla_z\log A$.
It can also be checked by differentiating the determinant of
$DT(z+\theta\chi qf)DT(z)^{-1}$ at zero. Differentiating $S_0(T(z))$
leaves the same log-Jacobian term and
$f\cdot(z-\bar z)/q$. Subtract them. Zero-row sets have zero density
measure; the weak identity already covers them. Finally use conditional
expectation against each retained descriptor test to obtain its score.
:::

:::{prf:remark} Degenerate and excluded density regimes
:label: rem-nyc-density-regimes

If $q=0$, there is no O innovation in the output. If $s=0$ and $q>0$,
conditioning on the full A2/terminal position array determines that O draw.
These give conditional point masses rather than the displayed density.
The actual fixed-seed convention likewise gives no Gaussian density.
An earlier absorbed row, enabled graph/Boris/metric force, donor-history
feedback, changed cap, or changed B2 normalization changes the native map
and requires its own inverse calculation. These statements concern the
specific density formula; they are not failures of an emergent gauge theory.
:::

(sec-nyc-ray-connection)=
## 4. A local connection and curvature of the actual ray transport

:::{prf:theorem} Exact local ray connection and its native source transport
:label: thm-nyc-local-ray-connection

On the available real-coordinate domain $|x|^2+|u|^2>0$, the existing
normalized ray-overlap links have the local one-form and curvature

$$
\begin{aligned}
A&=-i r^\dagger dr=\frac{x\cdot du-u\cdot dx}{R^2},
                 &&R^2=|x|^2+|u|^2,\\
F=dA&=\frac{2\sum_a dx_a\wedge du_a}{R^2}
 -\frac{2(x\cdot dx+u\cdot du)\wedge(x\cdot du-u\cdot dx)}{R^4}.
\end{aligned}
$$

For the actual endpoint-rephasing convention $r_e'=e^{i\alpha_e}r_e$,
$\ell_{ab}'=e^{i(\alpha_b-\alpha_a)}\ell_{ab}$,
$A'=A+d\alpha$, $F'=F$, and every recorded closed-loop holonomy is
unchanged. Reversing an available edge conjugates its link.
These are local transformations of the ray representative; no invariance of
the gas sampling law under physical changes of $(x,u)$ is presumed.

For a $C^2$ ray map on a parameter chart, the normalized overlap along a
small edge is $1+iA(\delta)+O(|\delta|^2)$. Its infinitesimal oriented
closed-loop phase is $F$ evaluated on the loop's area, with the orientation
of the ordered overlaps. This assertion applies to a chart of the actual
record map; smoothness of an unknown spacetime field is not an input.

At fixed terminal positions $x=y$, the curvature is the explicit native
conditional two-form

$$
F^y=-\frac{2(u\cdot du)\wedge(y\cdot du)}{(|y|^2+|u|^2)^2}.
$$

In the regime of {prf:ref}`thm-nyc-explicit-capped-action`, its weak source
transport is obtained by the actual pushforward $X_f$:

$$
\partial_f O=DO\,[X_f],\qquad
\int\partial_f O\,p_0^{\mathcal H,y}du
=\int O M_\perp\,p_0^{\mathcal H,y}du
$$

for bounded smooth available-link cylinder tests whose pullback has bounded
first derivative. For a link with overlap $Q_{ab}=r_a^\dagger r_b\ne0$,
its phase derivative is specifically

$$
\partial_f\arg\ell_{ab}
=\Im\frac{(\partial_f r_a)^\dagger r_b+r_a^\dagger\partial_f r_b}
                       {r_a^\dagger r_b}.
$$

Masks crossing either configured threshold use the exact bounded-measurable
source response and retain their boundary contribution; a branch derivative
alone is not their full likelihood derivative.
:::

:::{prf:proof}
Let $z=x+iu$ and $R^2=z^\dagger z$. Then
$z^\dagger dz=x\cdot dx+u\cdot du+i(x\cdot du-u\cdot dx)$.
The derivative of normalization cancels the real term in $r^\dagger dr$,
giving $A$. Differentiating its numerator gives
$2\sum_a dx_a\wedge du_a$, and differentiating $R^{-2}$ gives the
second term of $F$. Rephasing gives
$-i(r')^\dagger dr'=-ir^\dagger dr+d\alpha$ and the endpoint link law.
Their factors telescope around the actual ordered boundary.

For a chart edge, Taylor expand $r(t+\delta)$ and its overlap with $r(t)$.
Its first derivative is $r^\dagger dr=iA$, a purely imaginary number,
so dividing by its modulus leaves that first-order term. To identify the
curvature, sum the first-order one-form around a shrinking rectangle and
expand its coefficients at the common base point. The constant terms
cancel, leaving $(\partial_1A_2-\partial_2A_1)\delta_1\delta_2$;
refining triangular boundaries gives the corresponding oriented area.
This proves the asserted local chart coefficient. Put $dx=0$ to obtain
$F^y$. Pull back any specified smooth cylinder through the actual ray and
cap maps, and apply the preceding weak identity. Differentiating the
argument of a nonzero complex number gives $\Im(dQ/Q)$, proving the
displayed link variation. For a discontinuous availability indicator use
the exact density-ratio differentiation for bounded measurable functions;
it includes translations across its threshold by definition.
:::

:::{prf:corollary} A positive conditional curvature regime of the actual readout
:label: cor-nyc-native-ray-curvature

For $d\ge2$, $V>1$, $y=e_1$ and $u=e_2$ belong to an actual positive-density
velocity chart whenever the inverse test holds. On that chart

$$
F^y(\partial_{u_1},\partial_{u_2})=\frac12.
$$

Every sufficiently small open neighborhood of this pair has a nonzero
conditional curvature component and positive conditional probability.
The same conclusion holds after rotation and positive rescaling into a
positive velocity radius and an interior position neighborhood away from
zero, whenever its actual ray-availability inequalities hold.
For $d=1$, the ray fiber is a complex line: every available normalized
overlap loop is exactly one. For a fixed $d\ge2$ position $y\ne0$,
$F^y$ vanishes on the locus $u\parallel y$ and is nonzero off that locus.
This is the curvature of the actual $U(1)$ ray line; it is not an
$SU(3)$ curvature or an identified physical spacetime curvature.
:::

:::{prf:proof}
Substitute the two vectors in $F^y$; its wedge applied to
$(\partial_{u_1},\partial_{u_2})$ is $-1$, and the denominator is four.
Continuity preserves a strict lower bound on a small neighborhood, and
the explicit positive velocity density gives that neighborhood positive
probability. Rotation and positive rescaling preserve nonvanishing; the
declared availability test retains the configured numerical threshold.
In dimension one each unit ray is a scalar phase, whose
ratios telescope. For $y\ne0$, the wedge of $u\cdot du$ and $y\cdot du$
vanishes precisely when these real covectors are dependent.
:::

:::{prf:corollary} Nonconstant interaction holonomy at the unchanged reference
:label: cor-nyc-reference-positive-face

Use the real-coordinate initialization law of the unchanged
`RunConfig::viscous_euclidean()` dynamics: independent uniform positions
in $[-1,1]^3$ and zero entering velocities. Every dynamic and readout
parameter remains at its original value. A first-step ray interaction
triangle has positive probability of a strictly positive defect, and its
conditional phase is nonconstant on a positive-probability preparation
regime. This applies to the actual distance/clone interaction graph and
the one-step terminal-alive restriction.

The regime can be evaluated around the two tagged initial positions

$$
x_i=.5e_1,\qquad x_j=.5(e_1+e_2).
$$

Choose the actual reciprocal distance companions $i\to j$, $j\to i$,
and cloning companion $i\to j$. At these positions the two diversity
measurements coincide, while $R_i=-.125>R_j=-.25$. The actual reference
fitness satisfies $V_i>V_j$, so the configured gate of $i$ is persistence
with probability one. Every other donor and gate outcome is retained.
At the actual target

$$
Y_i=.5e_1,\qquad u_i=.5e_2,
$$

the triangle has
$H=(1-i)/\sqrt2$ and $w=1-1/\sqrt2>0$; at $u_i=0$ it has $H=1$.
Both targets and small neighborhoods are below the native cap and strictly
within all ray and overlap thresholds. This establishes actual native
loop nondegeneracy, not only a nonzero formal curvature at an arbitrary
chosen ray configuration. No uniform-in-$N$ constant or full color-sector
curvature is inferred from this finite event.
:::

:::{prf:proof}
Take all other initial positions in a small interior neighborhood of zero.
The prescribed pair assignments have positive probability under the
reference independent, nonself Gaussian companion samplers. Their common
diversity values follow from symmetry of the actual squashed algorithmic
distance, and both rows use the same completed population mean and
regularized standard deviation. The reward map is strictly increasing
and its exponent is one; the diversity factors are equal and positive.
Thus the stated reward inequality gives $V_i>V_j$. The actual gate
formula has $(V_j-V_i)_+=0$. By continuity and the positive normalization
floors, a small neighborhood of these finite positions keeps this strict
inequality and hence persistence. Its initialization probability is
positive. On this event the actual graph contains the triangle whose
pre events are $i,j$ and whose final event is the persisted slot $i$.
All entering velocities are zero; copying and any collision rotation
keep them zero, regardless of other accepted clones.

In the target $|u_i|=.5$, the cap inverse has length $2/3<1$.
Choose every other target final velocity within a small ball around zero.
The uniform target inverse theorem has $b_1>.9873$ for **every** actual
preparation, including any of the unbounded other-row clone jitters.
Its explicit joint density is positive on open neighborhoods of these
targets and any finite terminal position array. Choose the terminal
positions of the tagged events within the actual domain. Their eligible
marks then hold, and these neighborhoods have positive conditional
probability for every such preparation. The ray products at the two
displayed targets give their asserted holonomies. Continuity gives
disjoint neighborhoods of their phases. Each has positive probability
under the actual conditional density, proving a nonconstant conditional
phase. Integrating over the positive-probability initial/companion/gate
event gives the claim for the complete native law. One-step survival
normalization is positive and does not remove either interior event.
:::

(sec-nyc-native-face-action-limit)=
## 5. A native face-fluctuation action and its local first variations

:::{prf:definition} Existing finite-step families for the face-action limit
:label: def-nyc-face-limit-family

Keep every field of $\mathfrak P$ fixed except its existing timestep $h>0$,
and, only in the explicitly stated spatial comparison, the prescribed
initial positions. Fix a finite population, terminal boundary schedule,
$\gamma\ge0$, $b_O>0$, $\sigma_x>0$, $\rho>0$, $\nu\ge0$ and $V>0$.
Use either existing count or nonself row normalization. The potential is
one of the actual $C^1$ providers near the finitely many prepared positions,
including the quadratic and Styblinski--Tang providers. Its finite local
derivative bound is computed from that provider on those neighborhoods.
No law-level regularity hypothesis is introduced.

The entering velocities of this existing initial-law regime are all zero.
Condition on the actual complete companion, gate, jitter and collision
record before B1. It is denoted $\mathcal H_0$; its positions $X^J$ are
finite, and its collision velocities remain zero. Retain a finite list
of existing interaction triangles $e$. Triangle $e$ has pre-update events
at $x_{i(e)}\ne0$ and $x_{j(e)}\ne0$, and the final event of slot $i(e)$.
Require the **actual recorded** gate of this future slot to be persistence,
so $X^J_{i(e)}=x_{i(e)}$. Each pre-event is eligible and all these positions
lie in the interior of the terminal domain. The list and this persistence
event must have positive probability under the configured sampler and gate;
their actual probabilities are retained. Conditions on other clones and
revivals are not replaced by independent donor laws.

Put

$$
a_e=\frac{x_{i(e)}}{|x_{i(e)}|},\qquad
b_e=\frac{x_{j(e)}}{|x_{j(e)}|},\qquad
t_e=a_e\cdot b_e\ne0,\qquad
L_e=\frac1{|x_{i(e)}|}\left(a_e-\frac{b_e}{t_e}\right).
$$

The actual pre rays have zero imaginary part. For the ordered triangle
$(\text{pre }i,\text{final }i,\text{pre }j)$ let $\phi_{e,h}$ be the
principal holonomy phase. Its phase is zero at $h=0$, including when
$t_e<0$, because the two real signed overlaps have the same sign.
For the literal numerical readout require the limiting pre/post ray norms
and overlaps to strictly exceed its actual thresholds. These inequalities
hold for the evaluated example below. Condition the terminal positions at

$$
Y_h=m_h+\sqrt h\,y,
$$

where $y\in\mathbb R^{Nd}$ is any fixed finite array, and $m_h$ is the
actual prepared terminal center. These are fibers of the unchanged map;
the claim also holds locally uniformly for $y$ in compact sets. The source
is the existing O-stream innovation shift $\theta f$. In this family the
previous prepared record is the same at every $h$ and its force evaluations,
OU coefficient, cap and terminal diffusion are reevaluated at that $h$.
:::

:::{prf:theorem} Joint native face Gaussian, action and weak source variation
:label: thm-nyc-joint-native-face-action

In {prf:ref}`def-nyc-face-limit-family`, set

$$
\eta_{e,h}=\frac{\phi_{e,h}}{\sqrt h},\qquad
C_{e,(i,a)}=b_O\mathbf1_{i=i(e)}(L_e)_a,\qquad
\Gamma=CC^{\mathsf T}.
$$

At fixed $\mathcal H_0$ and the actual terminal-position fiber,
under the actual O source one has the joint limit, with every fixed
polynomial moment,

$$
\eta_h\ \Longrightarrow\ \eta_\theta=CZ+\theta Cf,
\qquad Z\sim N(0,I_{Nd}).
$$

The covariance is therefore the explicit native vertex-star matrix

$$
\Gamma_{ee'}=b_O^2\mathbf1_{i(e)=i(e')}L_e\cdot L_{e'}.
$$

Its support $\mathcal R=\operatorname{ran}C$ is retained even when
$\Gamma$ is singular. If $r=\operatorname{rank}\Gamma>0$, the limiting
native density relative to Euclidean $r$-dimensional volume on $\mathcal R$
is

$$
p_\theta^{\rm face}(\eta)
=\frac{\exp[-\tfrac12(\eta-\theta Cf)^{\mathsf T}
                     \Gamma^\dagger(\eta-\theta Cf)]}
 {(2\pi)^{r/2}\operatorname{pdet}(\Gamma)^{1/2}}.
$$

Thus the **native limiting action** and its source score are

$$
\begin{aligned}
S_0^{\rm face}(\eta)
 &=\tfrac12\eta^{\mathsf T}\Gamma^\dagger\eta
    +\tfrac r2\log(2\pi)+\tfrac12\log\operatorname{pdet}\Gamma,\\
j_f^{\rm face}(\eta)&=(Cf)^{\mathsf T}\Gamma^\dagger\eta,\qquad
X_f^{\rm face}=(Cf)\cdot\nabla_{\mathcal R},\\
\int X_f^{\rm face}O\,p_0^{\rm face}
 &=\int O j_f^{\rm face}\,p_0^{\rm face},\qquad
\operatorname{div}_{\mathcal R}X_f^{\rm face}=0.
\end{aligned}
$$

For bounded $C^1$ face tests with bounded derivative, these are also the
limits of the actual finite-$h$ source responses. The face Fisher information
is $f^{\mathsf T}C^{\mathsf T}\Gamma^\dagger Cf\le|f|^2$.
The original measured defects obey jointly, with every fixed moment,

$$
\frac{w_{e,h}}h\Longrightarrow\frac{\eta_{e,0}^2}2.
$$

All other preceding likelihood factors are retained in the mixing law of
$\mathcal H_0$. The result identifies the conditional fluctuation action;
an unconditional mixture or a geometry-only posterior is not replaced by
this one Gaussian. In particular this calculation discharges the native
source/face-action correspondence for this existing local $U(1)$ readout,
without changing the full gas action into a Wilson sampling rule.
:::

:::{prf:proof}
For the zero prepared velocities the first kick is
$v_1=aF_{
m pot}(X^J)$: the actual initial viscous force is zero.
Consequently A1 and the prepared center give
$p_h=X^J+O(h^2)$ and $m_h=X^J+O(h^2)$, for the fixed finite preparation.
The constants in this statement are its actual finitely evaluated
potential values. The conditional O law from the two-noise calculation is

$$
\xi=\frac{aq_h}{\tau_h^2}(Y_h-m_h)
        +\theta\chi_h f+\sqrt{\chi_h}\,Z.
$$

Here $q_h/\sqrt h\to b_O$, $\chi_h\to1$, and the first term is
$O(h)y$, because $\tau_h^2=\sigma_x^2h+O(h^3)$.
These limits are locally uniform in $y$. Each B2 position is
$X^J+O(h^2)+O(h^{3/2}\xi)$. On bounded innovation sets the actual
potential thus remains bounded by its finite local provider profile.
The count viscosity and row viscosity are both bounded in magnitude by
$2\nu\max_i|z_i|$, since their respective total row weights are at most
one and exactly one. A row with no nonself site uses its existing singleton
convention and has zero viscous increment. It follows that the full B2
velocity, before the cap, is

$$
w_{i,h}=b_O\sqrt h\,(Z_i+\theta f_i)+O(h)(1+|Z|^2)
$$

on a fixed local provider neighborhood. The actual radial cap differs from
the identity there by $O(|w_{i,h}|^2/V)$, so the final velocity has the same
leading term. The conditioned terminal position of a persisted slot is
$x_i+O(\sqrt h)$.

For the fixed pre rays $a_e,b_e$ and final ray $r_{i,h}$, normalization by
a positive real number does not alter the phase of either overlap. Hence

$$
\phi_{e,h}
=\arg\{[a_e\cdot(Y_{i,h}+iu_{i,h})]
       [(Y_{i,h}-iu_{i,h})\cdot b_e]t_e\}.
$$

At $Y_i=x_i$ and $u_i=0$, its bracket is the positive number
$|x_i|^2t_e^2$. Its first derivative in the real position is zero.
Its first derivative in velocity is precisely
$L_e\cdot du_i$. Taylor expansion on this available-link neighborhood
therefore gives

$$
\frac{\phi_{e,h}}{\sqrt h}
=b_OL_e\cdot(Z_{i(e)}+\theta f_{i(e)})
   +O(\sqrt h)(1+|Z|^4+|y|^4).
$$

To justify moments without bounding the algorithm's Gaussian noise, restrict
only the proof estimate to $|Z|\le h^{-1/8}$ for sufficiently small $h$.
There all evaluated positions remain in the same provider neighborhood,
all relevant denominators stay separated from zero, and the polynomial
Taylor remainder is valid. For any finite power, its expectation tends to
zero by the actual Gaussian moments. On the complement, the true principal
phase has modulus at most $\pi$ and its unavailable tagged component has
the zero extension with the retained mask; the Gaussian tail is bounded by a constant times
$e^{-c h^{-1/4}}$. This dominates every power of $h^{-1}$.
The same estimate controls changes of the strict availability masks and
the terminal tag mask, since $Y_{i,h}\to x_i\in D^\circ$.
It proves joint convergence and every fixed moment. No innovation is
removed from the law. The cosine Taylor remainder is at most
$|\phi|^4/24$, giving $w/h\to\eta^2/2$ with moments by the same argument.

Diagonalize $\Gamma$ on its positive-eigenvalue subspace. The orthogonal
projection of $CZ$ onto that subspace has independent normal coordinates
with those variances; integrating its density gives the displayed
$r$-dimensional density and action. Its score follows by differentiating
that density in the supported mean direction $Cf$. Integration by parts
in these real normal coordinates gives the weak identity and zero
divergence. The matrix $C^{\mathsf T}\Gamma^\dagger C$ is the orthogonal
projection onto $\operatorname{ran}C^{\mathsf T}$, proving the Fisher bound.

For the original finite-$h$ source response, the exact conditional Gaussian
score is $f\cdot(\xi-\ell_h)$. It converges in every fixed moment to
$f\cdot Z$. Bounded face tests and the proved moment convergence imply
that their responses converge to $E[O(CZ)f\cdot Z]$.
The Gaussian decomposition of $Z$ into its projection onto
$\operatorname{ran}C^{\mathsf T}$ and its independent orthogonal
complement gives
$E[f\cdot Z\mid CZ]=(Cf)^{\mathsf T}\Gamma^\dagger CZ$.
This is the displayed limiting action score. Dominated differentiation
also holds locally uniformly for finite $\theta$, using the Gaussian
exponential likelihood. Finally, averaging over the actual preceding
record gives its original mixture, rather than a prescribed new prior.
:::

:::{prf:corollary} The actual weighted-Wilson matching regime
:label: cor-nyc-wilson-matching-regime

In {prf:ref}`thm-nyc-joint-native-face-action`, let $B$ be the diagonal
matrix of the actual configured face coefficients $\beta_e$. The native
limiting action energy equals the limit of the existing scaled weighted
Wilson readout $h^{-1}\sum_e\beta_e w_{e,h}$ exactly when

$$
\eta^{\mathsf T}(\Gamma^\dagger-B)\eta=0
\quad\text{for every }\eta\in\operatorname{ran}C.
$$

This condition is evaluated from the primitive pre positions, $b_O$ and
the actual readout weights. A proved positive regime is a list with one
noncollinear available face at each distinct future slot, when its actual
retained coefficients satisfy $\beta_e=[b_O^2|L_e|^2]^{-1}$.
This is a test on the existing coefficients, not a replacement readout.
Its covariance is diagonal and positive,
so this regime gives the action and every weak first variation of that
limiting Gaussian. A single face requires only its scalar coefficient.

For the existing three-dimensional reference choose the legitimate entering
positions $x_i=.5e_1$, $x_j=.5(e_1+e_2)$, zero entering velocities, and
the actual persistence/companion event in
{prf:ref}`def-nyc-face-limit-family`. These points are strictly within
$D=[-2,2]^3$, their pre overlap is $1/\sqrt2$, and

$$
L_e=-2e_2,\qquad b_O^2|L_e|^2=4,\qquad
\frac{w_{e,h}}h\Longrightarrow2Z_2^2,
\qquad S_0^{\rm face}(\eta)=\frac{\eta^2}{8}
                                      +\frac12\log(8\pi).
$$

Thus the literal experiment-24 unit-weight face readout has native action
energy one quarter of its scaled defect. Its coefficient is derived from
the actual noise and geometry. It is not obtained by changing the sampling
law. Existing action-unit calibration can record this scalar conversion;
a common scalar cannot fix unequal variances of a general many-face list.

There is also an explicit regime for the **original unit coefficient**, so
no changed weight or action calibration is needed: take
$x_i=.5e_1$, $x_j=.5e_1+.25e_2$, $b_O=1$, and the same actual
reciprocal-companion/persistence event. These are permitted initial state
coordinates strictly inside the reference initialization box. Then
$L_e=-e_2$, $\Gamma_{ee}=1$, and

$$
\frac{w_{e,h}}h\Longrightarrow\frac{Z_2^2}2,
\qquad S_0^{\rm face}(\eta)=\frac{\eta^2}2+\frac12\log(2\pi).
$$

Thus a literal existing ray-face observable has a proved native quadratic
action and first-variation limit with the unit Wilson coefficient in this
specified initial-coordinate regime. This is a conditional one-face
identification; a random initialization mixture retains its varying $L_e$.

For two faces sharing a future slot with linearly independent vectors
$L_1,L_2$ and $L_1\cdot L_2\ne0$, the $2\times2$ covariance is positive
with a nonzero off-diagonal entry. Its inverse has a nonzero mixed entry.
No diagonal face-weight matrix then equals its action quadratic form.
The native limiting action nevertheless remains local to each future
vertex star, with its proved block covariance and all its cross terms.
:::

:::{prf:proof}
The preceding theorem gives the two quadratic forms on the same supported
limit, so subtraction gives the necessary and sufficient identity on that
support. For distinct future tags the rows of $C$ have disjoint innovation
supports; their covariance is diagonal with the displayed positive entries.
Choosing the stated existing coefficient gives its inverse. Substitute the
two prototype positions to obtain $a=e_1$, $b=(e_1+e_2)/\sqrt2$,
$t=1/\sqrt2$ and $L=-2e_2$. All norm and overlap thresholds are separated
from their boundary. The scalar action and defect limit follow. For the
second prototype $b/t=e_1+.5e_2$ and $|x_i|=.5$, so $L=-e_2$ and its
variance is one. Its reward is again lower than that of $i$, while
reciprocal distance companions have equal diversity values; the actual
clone-$j$ gate at $i$ is therefore persistence, exactly as in
{prf:ref}`cor-nyc-reference-positive-face`.
For two linearly independent $L$ vectors the Gram determinant is positive.
The inverse of their covariance has mixed entry
$-\Gamma_{12}/\det\Gamma\ne0$, so a diagonal form cannot agree on
its full two-dimensional range. The stated block locality follows from
the explicit indicator in $\Gamma_{ee'}$.
:::

(sec-nyc-spatial-time-regime)=
## 6. Actual spatial and temporal scaling of this connection

:::{prf:theorem} Shrinking-edge curvature scaling and its primitive noise regimes
:label: thm-nyc-shrinking-edge-curvature

Use the existing family in {prf:ref}`def-nyc-face-limit-family` for a single
persisted future slot, with $x_i=x\ne0$ and
$x_j=x+\epsilon r$, $\epsilon\downarrow0$. Retain its actual eligible
companion/persistence event and strict availability masks. Use the same
fixed finite consumed donor/gate/jitter/collision record in this initial
coordinate family; its prepared positions are then locally bounded by
the actual initial coordinates and those finite jitters. Suppose
$P_xr\ne0$, where $P_x=I-xx^{\mathsf T}/|x|^2$.
This is a derived geometric test on the actual initial coordinates.
For any joint $h\downarrow0$, $\epsilon\downarrow0$ with the actual
prepared finite positions locally bounded, the fixed-geometry conditional
phase has the leading noise coefficient

$$
\frac{\phi_{e,h}}{\epsilon\sqrt h}
\Longrightarrow -\frac{b_O}{|x|^2}(P_xr)\cdot Z_i,
\qquad
\operatorname{Var}\!\left(\frac{\phi_{e,h}}
                              {\epsilon\sqrt h}\right)
\longrightarrow v_x=\frac{b_O^2|P_xr|^2}{|x|^4}>0.
$$

Consequently the classical area-normalized quantity
$\phi_{e,h}/(\epsilon h)$ is not tight for fixed $b_O>0$.
This is a conclusion about the actual ray-curvature coefficient on
algorithmic-time faces, not a failure of a quantum distributional field.

The existing parameter family $b_O(h)=\bar b\sqrt h$ instead gives

$$
\frac{\phi_{e,h}}{\epsilon h}
\Longrightarrow
-\frac{P_xr\cdot F_{\rm pot}(x)}{|x|^2}
-\frac{\bar b}{|x|^2}(P_xr)\cdot Z_i.
$$

The actual force is used in this expression; both kicks contribute its
full coefficient. For the quadratic reference,
$F_{\rm pot}(x)=-\lambda x$, its drift term is zero.
For $b_O(h)=o(\sqrt h)$ the limit is the displayed deterministic
force-induced coefficient. These parameter families change only the
existing OU-amplitude field and retain its actual update. They do not
claim that its single-face limit survives temporal averaging as a nonzero
continuum quantum field.
:::

:::{prf:proof}
The derivative of $b=(x+\epsilon r)/|x+\epsilon r|$ at zero is
$P_xr/|x|$, while $a\cdot b=1+O(\epsilon^2)$.
Thus

$$
L_e=-\epsilon\frac{P_xr}{|x|^2}+O(\epsilon^2).
$$

The actual face phase as a function of final $(Y_i,u_i)$ vanishes
identically when $\epsilon=0$, for every available post ray, because
the two pre rays then coincide. Its Taylor remainder therefore has
an additional factor $\epsilon$. More explicitly, differentiating
the product in the previous proof once in $\epsilon$ and then in
$(Y_i-x,u_i)$ on a fixed neighborhood with nonzero overlaps gives
a bounded mixed derivative. Hence the previous moment estimates yield
an error $O(\sqrt h+\epsilon)$ in
$\phi/(\epsilon\sqrt h)$, locally uniformly in the prepared bounded
coordinates. They hold for every joint schedule. Gaussian tails treat
the complement without replacing the draw distribution.

The limiting centered normal variable has a continuous density and positive
variance. For any fixed $M$, the probability that
$|\phi/(\epsilon h)|\le M$ is bounded by the probability that
$|\phi/(\epsilon\sqrt h)|\le M\sqrt h$. Its limit superior is at most
the probability that the nondegenerate limit is zero, which is zero.
Thus the family is not tight. This also avoids inferring nontightness
solely from diverging second moments.

If $b_O(h)=\bar b\sqrt h$, the actual OU velocity has stochastic
leading part $h\bar b Z_i$ instead. The first kick gives
$aF_{\rm pot}(x)$, the OU attenuation is $1+O(h)$, and the second
kick gives another $aF_{\rm pot}(x)+o(h)$. The prepared viscous force
is zero, and the B2 viscous force is $O(h)$, so its kick is $O(h^2)$.
The cap differs from the identity by $O(h^2)$ in this regime.
The conditioned terminal-position residual mean is $o(h)$ in velocity
units, as follows by substituting $q_h=O(h)$ in the exact conditional
normal law. The same differentiated phase expansion now gives the
two displayed contributions. With $b_O=o(\sqrt h)$ the stochastic
part is $o(h)$. Orthogonality $P_xr\cdot x=0$ proves the quadratic
reference assertion.
:::

:::{prf:remark} Remaining non-Abelian and full-field identification
:label: rem-nyc-nonabelian-residual

The proofs above advance the native action correspondence in three concrete
ways: an explicit capped B2 geometry-fiber density at the unchanged count
reference, its actual velocity/link first variations, and a limiting
interaction-face action with evaluated covariance and Wilson coefficients.
They use the same algorithmic likelihood and actual ray transports.

The local connection identified here is $U(1)$. The full color Gram and
determinant hierarchy and the nonzero determinant innovation proved in
{prf:ref}`thm-ym-native-physical-gauge-hierarchy` and the native determinant
chapter are different retained channels. This ray connection does not
identify their $SU(3)$ spacetime transport. The attribution $SU(2)$
connection requires its actual prescribed IG/IA transport maps; mere group
membership of an undecoded payload does not supply its native coefficients.
Conversely, an optional absent stored Rust link array does not preclude a
mathematical pushforward connection of the full executed history.

The unresolved central step is to derive a non-Abelian comparison transport
for those same native channels, with its spatial/temporal scaling, and to
prove that the corresponding descriptor likelihood, including the full
preparation mixture and its geometry posterior, has the target Yang--Mills
action and first variations. The proved vertex-star covariance shows the
cross terms that such an identification must account for. Extending a
single conditional small-step action to the full time-dependent record also
requires control of persistence failures, actual cloning jumps, changing
geometry and the varying native preparation law. No unknown local gauge
invariance, chaos, curvature regularity or native action convergence has
been installed as an extra premise.
:::
