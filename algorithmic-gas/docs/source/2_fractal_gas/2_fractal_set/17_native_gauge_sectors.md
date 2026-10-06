# Native gauge descriptor sectors and their actual transfer correspondence

(sec-npg-ledger)=
## 1. The existing projector and color-orbit dictionaries

:::{prf:definition} Complete gauge-sector execution and observation data
:label: def-npg-complete-record

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`, with every recursive
algorithm, landscape/provider, initial-law, schedule, noise, arithmetic,
allocation/error, recording, mask and physical-calibration field. The
finite stochastic proofs use the actual real-coordinate independent-Gaussian
execution convention. A deterministic fixed seed has its different actual
law. No new random field or gauge projection is added. The algebraic inverse and
density formulas below belong to that real-arithmetic branch. Floating-point
projection, rounding and fixed-stream execution retain their actual
pushforward laws; no exact inverse or Gaussian density is asserted for their
rounded finite-state output.

The first dictionary is a primitive projector refinement of the existing
phase-space-ray readout in
`algorithmic-gas/crates/algorithmic-gas/src/physics/qft/run_observables.rs`,
experiments $1,11,15,23,24,29$. At each retained pre/final event it already
constructs

$$
r_i=\frac{x_i+iv_i}{\sqrt{|x_i|^2+|v_i|^2}}
$$

in the actual numeric component frame. Its availability indicator $a_i$
requires the original eligible mark and norm greater than
$\delta_r=10^{-12}$; normalized overlap availability uses the separate
$10^{-10}$ threshold. Event epoch, step, slot, generation and version are
retained. Its existing primitive phase quotient is

$$
P_i=r_ir_i^\dagger,
$$

extended by zero only when the original ray is unavailable. The dictionary
$D_{\rm ray}$ retains the actual positions, validity/event labels,
availability indicators and matrix entries of these projectors. Matrix
entries are retained in the ambient component frame used by the numerical
readout. This is the measurable primitive refinement of the executed rays;
it is not asserted to be recoverable from the literal final averaged loop
report. Computing $P_i$ from an already used ray adds no sampled trajectory.
Absolute event labels align recorded rows; the stationary clock formulas
use the native current-state core, rather than adding an ever-increasing
step label as a stationary coordinate. In the canonical passive, period-one
restriction of {prf:ref}`def-npt-complete-record`, that core is the position,
velocity and status arrays. If a variant consumes additional memory or
dynamic coordinates, their original values must also be retained before
claiming recovery of its complete state.
The local representative change $r_i\mapsto e^{i\alpha_i}r_i$ leaves each
$P_i$ invariant. The existing normalized overlap loops remain functions of
these ray lines wherever their original overlap thresholds hold.

The second dictionary is the different direct common-$SU(3)$ color-orbit
descriptor of {prf:ref}`thm-sm-direct-orbit-isomorphism`: full complex Gram
coordinates $q_{ij}$, determinants $b_{ijk}$, their conjugates, triangle
products $\Pi_{ijk}$, masks and the declared geometry coordinates.
Projector traces such as $\operatorname{Tr}(c_ic_i^\dagger c_jc_j^\dagger
c_kc_k^\dagger)$ belong to this dictionary. The individual color-projector
matrix entries in a component frame are gauge-covariant and are not
identified with their common-$SU(3)$ orbit quotient. The ray projectors and
the color projectors also consume different recorded vectors.

Clock transfer below uses the already declared calibration
$a_{\rm phys}=t_*h>0$ and the same native update clock as
{prf:ref}`def-npt-complete-record`. The separate four-position-coordinate
physical reflection of {prf:ref}`prop-ym-recorded-physical-transformations`
retains its own embedding and support convention. No clock step is silently
substituted for its $x^0$ coordinate.
:::

(sec-npg-ray-determining)=
## 2. What the actual ray-projector descriptor determines

:::{prf:theorem} Exact kinetic recovery from a native ray projector and its position
:label: thm-npg-ray-kinetic-inverse

For $d\ge2$, take an originally available ray, with its actual $x,v$ and
$P=rr^\dagger$. Set $A=\Re P$, $B=\Im P$, and $b=Bx$.
On the regular chart $b\ne0$ the actual velocity is recovered exactly by

$$
v=\frac{|x|^2 A Bx}{|Bx|^2}.
\tag{NPG.1}
$$

Thus positions and the primitive $U(1)$-invariant projector identify the
complete position/velocity row on this chart, including the velocity
component parallel to its position. Their recovery map is smooth on each
chart $|Bx|\ge\eta>0$. The excluded geometric locus is exactly
$x=0$ or $v\parallel x$; no information about the velocity magnitude is
claimed there. For $d=1$ this recovery regime is empty.

For the actual eligible norm-unavailable branch, the original condition
$\sqrt{|x|^2+|v|^2}\le\delta_r$ implies $|v|\le\delta_r$.
An ineligible row has its original missing projector and does not inherit
this small-velocity bound. All eligibility and norm masks remain explicit.
:::

:::{prf:proof}
Put $s=|x|^2+|v|^2$. Multiplying the actual normalized ray gives

$$
A=\frac{xx^{\mathsf T}+vv^{\mathsf T}}s,
\qquad B=\frac{vx^{\mathsf T}-xv^{\mathsf T}}s.
$$

For $x\ne0$, write $v=\alpha x+w$ with $x\cdot w=0$.
Then $Bx=|x|^2w/s$ and

$$
A Bx=\frac{|x|^2|w|^2}{s^2}v,\qquad
|Bx|^2=\frac{|x|^4|w|^2}{s^2}.
$$

Division proves (NPG.1) when $w\ne0$. The map consists of rational
operations with nonzero denominator on the chart, proving its smoothness.
If $x=0$ or $v\parallel x$, the displayed skew matrix vanishes. Conversely
$Bx=0$ for $x\ne0$ implies $w=0$. The unavailable-eligible bound follows
from the original sum of squared norms. Ineligibility is a different branch
and supplies no such norm inequality.
:::

:::{prf:lemma} Regular charts have full geometric measure under the proved native smoothing
:label: lem-npg-native-projector-charts

Use the real-coordinate canonical instances whose full final position and
velocity law on each nonempty terminal-mask stratum is absolutely continuous
as proved in {prf:ref}`thm-cgd-phase-smoothing` and its stated count/row
extensions. This includes the unchanged reference and its identified QSD
and Doob laws. For $d\ge2$, each available eligible row satisfies
$\Im(P_i)x_i\ne0$ almost surely. Thus all eligible available rows are
recovered simultaneously at fixed finite $N$, outside the actual
norm-unavailable branch.

On the all-eligible stratum, if missing norm rays are assigned reconstructed
velocity zero, the normalized mean-square reconstruction satisfies

$$
\frac1N\sum_i|v_i-\widehat v_i(D_{\rm ray})|^2\le\delta_r^2
\quad\text{almost surely}.
\tag{NPG.2}
$$

On the full marked state, retain the actual cap $|v_i|<V$ and set missing
ineligible rows to zero. The corresponding bound is

$$
\frac1N\sum_i|v_i-\widehat v_i|^2
\le\delta_r^2+V^2\frac{\#\{i:\text{ineligible}\}}N.
\tag{NPG.3}
$$

This keeps the discrete statuses that continuous gradients alone do not
control. It does not assume that the final state is all alive.
:::

:::{prf:proof}
For fixed $x\ne0$, its parallel velocities form a one-dimensional line in
$\mathbb R^d$, which is Lebesgue-null when $d\ge2$. The set $x=0$ is also
null. Fubini therefore makes the irregular locus null in the joint
position/velocity volume. The proved absolute continuity on every mask
stratum transfers this statement to the native transition law. The QSD
identity $\nu=\alpha^{-1}\nu Q$ transfers it to the QSD; multiplication by
its positive bounded eigenfunction transfers it to the Doob law. A finite
union over rows remains null. On every available eligible row the inverse
is exact. Every unavailable eligible row has error at most $\delta_r$ by
the preceding theorem. The original final cap bounds the error on an
ineligible row by $V$. Sum the squares to obtain (NPG.2)--(NPG.3).
:::

:::{prf:corollary} Native observable recovery on the determining chart
:label: cor-npg-projector-observable-recovery

Let $\mathcal R$ be the actual state chart where all retained rows are
eligible, rays are available, and $\Im(P_i)x_i\ne0$. Every bounded
measurable native core-state observable supported in $\mathcal R$ is a
measurable cylinder of $D_{\rm ray}$. In the stated canonical restriction
this is the full dynamical state. For an instance with additional consumed
coordinates the assertion concerns the recovered position/velocity/status
core, unless those coordinates are also included in the dictionary. No new
mode assignment is needed.
If a bounded current-state observable $f$ is Lipschitz in the normalized
velocity distance with its evaluated constant $L_f$, then on the
all-eligible stratum

$$
|f(x,v)-f(x,\widehat v(D_{\rm ray}))|\le L_f\delta_r.
\tag{NPG.4}
$$

The full marked version has bound
$L_f[\delta_r+V\sqrt{N^{-1}\#\{\text{ineligible}\}}]$.
The actual law, marks, initial condition, coordinate units and all consumed
parameters remain in the same observable and its evaluated Lipschitz
constant. No claim that arbitrary discontinuous observables obey (NPG.4)
is made.
:::

:::{prf:proof}
The regular chart is itself recognizable from $D_{\rm ray}$, and (NPG.1)
gives its measurable core-state inverse. Composing a bounded measurable
observable with this inverse on the chart and extending by zero outside it
gives its descriptor cylinder. The Lipschitz estimates follow by taking
square roots of (NPG.2)--(NPG.3). The triangle inequality gives the displayed
full-state square-root bound.
:::

(sec-npg-native-ray-reflection)=
## 3. A reflection calculation within the actual projector algebra

:::{prf:theorem} The interacting reference cycle survives on regular native projector charts
:label: thm-npg-reference-projector-cycle

Use the unchanged real-coordinate count reference, including its radial
cap, all donor/gate/collision patterns, final position diffusion and
terminal absorption. Keep the actual primitive $D_{\rm ray}$ above and
its stationary Doob law. For three all-alive consensus states
$x_i=0$, $v_i=ue_1$ define $g(u)=Vu/(V-u)$ and the original kinetic
coefficient

$$
\kappa_{\rm cap}
=\frac{[c-a^2\lambda(1+c)]/q^2-a^2/s^2}{(1-a^2\lambda)^2}.
$$

At velocities $r,2r,3r$, where $r=1/4$ and $V=2$, the actual forward
and reverse cycle densities have log ratio

$$
\begin{aligned}
\mathcal L_N
&=N\kappa_{\rm cap}r\{g(3r)+g(r)-2g(2r)\}\\
&=N\kappa_{\rm cap}
\frac{2V^2r^3}{(V-3r)(V-2r)(V-r)}
=.9139855912\ldots N>0.
\end{aligned}
\tag{NPG.5}
$$

There exist three disjoint open state cells $A_1,A_2,A_3$, each of positive
actual stationary probability, entirely inside regular available ray charts
and the all-alive stratum, for which the stationary cycle of their
conditional transition probabilities is strictly unequal to its reverse.
Their indicators are actual $U(1)$-invariant primitive projector/position
cylinders. Consequently there is a pair of those descriptor cylinders for
which the actual one-step stationary clock-reflection matrix is not Hermitian.

This result tests the actual primitive projector refinement of the executed
ray instrument. It is not a test of arbitrary invisible state modes, nor of
a freely assigned CAR walker index. It does not identify that instrument
with the distinct common-$SU(3)$ color orbit, with the literal averaged
loop report, or with the separate four-position-coordinate physical cut.
:::

:::{prf:proof}
The density formula and open-set continuity proof of
{prf:ref}`thm-npt-reference-consensus-cycle` retain every unbounded cloned
preparation using {prf:ref}`lem-npt-reference-local-density`. Its formula
(NPT.30) holds for each positive consensus velocity below the actual cap.
The mean-noise cross term is $N\kappa_{\rm cap}u g(v)$; all source-only,
target-only and cap-Jacobian terms cancel around a closed cycle. For
$r,2r,3r$ the remaining term is
$N\kappa_{\rm cap}r[g(3r)+g(r)-2g(2r)]$. Since
$g(u)=V^2/(V-u)-V$, bringing these three fractions to a common denominator
gives (NPG.5). The unchanged reference has
$\kappa_{\rm cap}=23.9921217696\ldots>0$.

Each consensus state has velocity norm at least $r>\delta_r$, so its rays
are already norm-available. Their collinearity with $x=0$ is not ignored.
The strict cycle persists on open neighborhoods by the proved actual
kernel-density continuity. Regular ray charts are dense there: perturb
each position by an arbitrarily small component perpendicular to its
nonzero velocity, without changing eligibility or ray availability.
The available common-part certificate may be chosen with $L_0>0$ and
$r_v>3r$, so small all-alive neighborhoods lie in its support. The actual
Doob stationary law has a locally integrable density $\varrho>0$ almost
everywhere there. Its eigenfunction obeys $m_N\le e_N\le1$. Thus $e_N$,
$\varrho$ and $\varrho/e_N$ have simultaneous Lebesgue differentiation
points of full measure. Choose three regular centers $s_i$ that are such
points inside the open strict-cycle product. This uses neither pointwise
eigenfunction continuity nor density continuity of the stationary law.

Their regular property is open. Let $A_i(\varepsilon)$ be disjoint
Euclidean balls of radius $\varepsilon$ about these centers inside the
all-alive regular chart, and write $q(s,t)$ for the actual killed density.
Its already proved local continuity gives

$$
\begin{aligned}
T_{ij}(\varepsilon)
&=\frac1{\pi(A_i)}\int_{A_i}\pi(ds)P(s,A_j)\\
&=\frac{1}{\alpha_N\int_{A_i}\varrho(s)\,ds}
\int_{A_i}\frac{\varrho(s)}{e_N(s)}
\left[\int_{A_j}q(s,t)e_N(t)\,dt\right]\,ds,\\
\lim_{\varepsilon\downarrow0}
\frac{T_{ij}(\varepsilon)}{|A_j(\varepsilon)|}
&=\frac{q(s_i,s_j)e_N(s_j)}{\alpha_Ne_N(s_i)}.
\end{aligned}
$$

To justify the limit, replace $q(s,t)$ by $q(s_i,s_j)$. Its error is at most
its uniform oscillation on $A_i\times A_j$, times the two displayed
nonnegative integrals. After normalization the ratios of their ball averages
are finite and converge at the selected differentiation points. The
oscillation tends to zero. The differentiation identities then give the
last line.

Eigenfunction ratios and $\alpha_N$ cancel around the limiting cycle, and
target volumes cancel exactly. The limiting forward/reverse ratio is the
strict raw-$q$ cycle at the selected centers. Hence for all sufficiently
small positive $\varepsilon$ the actual cells satisfy
$T_{12}T_{23}T_{31}>T_{21}T_{32}T_{13}$.
If every stationary pair flux were symmetric, then
$\pi(A_i)T_{ij}=\pi(A_j)T_{ji}$ for each pair; multiplying would contradict
this strict cycle. Some pair therefore has unequal flux. Its centered
indicator modes have unequal off-diagonal covariance coefficients exactly
as in {prf:ref}`cor-npt-reference-full-record-clock-reflection`.
All the indicators factor through $D_{\rm ray}$ by
{prf:ref}`cor-npg-projector-observable-recovery`. Their reflected matrix is
therefore a matrix within this actual gauge-projector observable algebra.
:::

:::{prf:theorem} A quantitative discrepancy for every positive transfer on those native gauge modes
:label: thm-npg-projector-positive-transfer-defect

Select the unequal-flux pair $A,B$ just proved. Let
$p_A=\pi(A)>0$, $p_B=\pi(B)>0$ and

$$
J_{AB}=\int_A\pi(ds)P(s,B)-\int_B\pi(ds)P(s,A)\ne0.
$$

The two actual centered descriptor functions
$f_A=\mathbf1_A-p_A$ and $f_B=\mathbf1_B-p_B$ have Gram matrix

$$
G=\begin{pmatrix}p_A(1-p_A)&-p_Ap_B\\
-p_Ap_B&p_B(1-p_B)\end{pmatrix},\qquad
\det G=p_Ap_B(1-p_A-p_B)>0.
$$

Define the genuine native isometric injection
$j:\mathbb C^2\to L^2_0(\pi)$ by
$jz=(f_A,f_B)G^{-1/2}z$. For every self-adjoint candidate transfer $B_0$,
including any proposed $B_0=e^{-a_{\rm phys}H}$ with $H\ge0$,

$$
\|Cj-jB_0\|
\ge\frac{|J_{AB}|}{2\sqrt{p_Ap_B(1-p_A-p_B)}}>0.
\tag{NPG.6}
$$

The native time-antisymmetric defect is thus visible on an explicitly
constructed isometric injection of actual gauge descriptor modes. It is not
an unverified hypothetical injection. These finite-population flux and
probability values are computed from the unchanged complete kernel and its
identified stationary law. Their positivity is proved above; no lower bound
uniform in population or cutoff is inferred from them.
:::

:::{prf:proof}
The cells are disjoint and the third cell has positive probability, so
$p_A+p_B<1$. Expanding centered indicator products gives the displayed
Gram matrix and its positive determinant. Hence the stated $j$ is an
isometry. Let $H_{ij}=\langle f_i,Cf_j\rangle$ for $i,j\in\{A,B\}$.
Its antisymmetric part is

$$
H-H^*=\begin{pmatrix}0&J_{AB}\\-J_{AB}&0\end{pmatrix}.
$$

The compression $T=j^*Cj$ has
$T-T^*=G^{-1/2}(H-H^*)G^{-1/2}$.
For a real symmetric two-by-two matrix $R$,
$R\left[\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\right]R
=(\det R)\left[\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\right]$.
Consequently $\|T-T^*\|=|J_{AB}|/\sqrt{\det G}$.
For $B_0=B_0^*$,
$\|T-T^*\|\le2\|T-B_0\|\le2\|Cj-jB_0\|$.
This proves (NPG.6) and retains any additional leakage out of the two-mode
space inside the actual discrepancy.
:::

(sec-npg-descriptor-transfer)=
## 4. The descriptor transfer, leakage and native decay

:::{prf:theorem} Native projector-sector transfer without a lumpability assumption
:label: thm-npg-descriptor-transfer-budget

For a current-state dictionary such as $D_{\rm ray}$ use the actual
stationary core law $\varrho_D=\pi$ and $\mathcal C_D=C$. For the
transition-attached B2 dictionary $D_{\rm col}$ use the actual augmented
law $\varrho_D=\widehat\pi$ and
$\mathcal C_D=\widehat C$ of {prf:ref}`def-npt-attached-record`. The
preceding passive record is part of that augmented state, even though it
does not drive the next update. A declared terminal recomputation on the
current state instead uses the first branch; stage alignment is retained.
Set $\mu=D_*\varrho_D$ and let
$j_D:L^2_0(\mu)\to L^2_0(\varrho_D)$ be the actual isometry
$j_Du=u\circ D$. Put $\Pi_D=j_Dj_D^*$ and

$$
T_D=j_D^*\mathcal C_Dj_D,\qquad
K_{D,n}=j_D^*\mathcal C_D^nj_D,\qquad
L_D=(I-\Pi_D)\mathcal C_Dj_D.
\tag{NPG.7}
$$

These are the actual one-step predictive descriptor operator, native
multi-time correlation operator and omitted conditional-prediction operator.
The existing primitive reference certificate gives the stage-specific bounds

$$
\begin{array}{lll}
D\text{ current-state}:&
\|T_D\|\le\sqrt{1-\delta_N},&
\|K_{D,n}\|\le(1-\delta_N)^{n/2},\\
D\text{ transition-attached}:&
\|T_D\|\le1,&
\|K_{D,n}\|\le(1-\delta_N)^{\lfloor n/2\rfloor/2}.
\end{array}
\tag{NPG.8}
$$

In particular the augmented transfer has
$\|\widehat C^2\|\le\sqrt{1-\delta_N}$; a marginal one-step
minorization is not used as a one-step bound for the attached record.

The descriptor compression need not reproduce the actual correlations by
its own powers. Their exact discrepancy satisfies

$$
\|\mathcal C_D^nj_D-j_DT_D^n\|\le n\|L_D\|,\qquad
\|K_{D,n}-T_D^n\|\le n\|L_D\|.
\tag{NPG.9}
$$

For any candidate self-adjoint positive descriptor transfer $B_0$,

$$
\|\mathcal C_Dj_D-j_DB_0\|^2
=\sup_{\|u\|=1}
\{\|(T_D-B_0)u\|^2+\|L_Du\|^2\}.
\tag{NPG.10}
$$

For the actual ray dictionary and bounded descriptor mode $u$, the regular
full-state inverse additionally gives

$$
\|L_Du\|\le2\|u\|_\infty\sqrt{\pi(\mathcal R^c)}.
\tag{NPG.11}
$$

This is a derived error on that actual dictionary; no hypothesis that the
chart complement is small is imposed. Its norm-unavailability and discrete
ineligibility parts retain their actual probabilities. For the counted
reference, $T_{D_{\rm ray}}$ is non-self-adjoint by the preceding theorem.
Therefore the actual one-step primitive ray prediction does not itself
supply a positive physical Hamiltonian, despite its proved native decay.
:::

:::{prf:proof}
Pushforward of the stationary measure makes $j_D$ isometric. Its adjoint
is conditional expectation onto the descriptor sigma algebra, so $\Pi_D$
is the orthogonal conditional-expectation projection. The actual full-law
covariance is $\langle j_Du,\mathcal C_D^nj_Dv\rangle$, proving the meaning of
$K_{D,n}$. Its contraction bounds follow by applying
{prf:ref}`thm-npt-minorization-l2` in the current-state case and
{prf:ref}`thm-npt-attached-two-step-decay` in the attached-record case,
before compressing.

The one-step identity is
$\mathcal C_Dj_D-j_DT_D=(I-\Pi_D)\mathcal C_Dj_D=L_D$. Telescope it through $n$ steps:

$$
\mathcal C_D^nj_D-j_DT_D^n
=\sum_{k=0}^{n-1}\mathcal C_D^{n-1-k}L_DT_D^k.
$$

Every outside factor is a contraction, giving (NPG.9). The two summands
$j_D(T_D-B_0)u$ and $L_Du$ are orthogonal, proving (NPG.10).
On the actual regular ray chart $\mathcal R$, the complete state is a
measurable function of $D$. Therefore $Cj_Du$ is descriptor-measurable
there. The chart is itself descriptor-measurable, so its conditional
expectation agrees with it there. On the complement both the predictor and
its conditional expectation have absolute value at most $\|u\|_\infty$.
This proves (NPG.11). The preceding actual descriptor reflected matrix
shows the compression's off-diagonal antisymmetric part is nonzero.
:::

(sec-npg-su3-hidden-centroid)=
## 5. A genuine unresolved kinetic fiber of the direct color orbit

:::{prf:theorem} Two trace-free centroid directions are invisible to the native color-orbit hierarchy
:label: thm-npg-su3-hidden-centroid

Retain the actual matched B2 three-component color with either Gaussian
count/row viscosity, its original threshold, and terminal position geometry
$Y$. Use the same actual conditioning as
{prf:ref}`thm-ngf-terminal-geometry-centroid`, with full preparation through
A1, $q,s>0$, $a=h/2$, and

$$
\tau^2=a^2q^2+s^2,\qquad\chi=s^2/\tau^2>0.
$$

Condition on the full preparation, $Y$, the relative force-input velocities
$r_i=z_i-N^{-1}\sum_jz_j$, and the trace of their centroid
$\mathbf1\cdot G$. Denote this actual sigma algebra by $\mathcal H_*$.
There are independent standard normals $Z_1,Z_2$, independent of
$\mathcal H_*$, and an orthonormal basis $e_1,e_2$ of
$\{u:\mathbf1\cdot u=0\}$ such that

$$
G=G_*+\frac{q\sqrt\chi}{\sqrt N}(e_1Z_1+e_2Z_2).
\tag{NPG.12}
$$

The complete common-$SU(3)$ orbit dictionary made of $q,b,\Pi$, their
conjugates, original color masks, terminal support masks and any declared
terminal-geometry/centroid-invariant weights is measurable with respect to
$\mathcal H_*$. Scalar readouts of retained preparation data also remain
measurable there. Thus it does not determine these two native Gaussian
kinetic directions.

In particular, for the actual B2 force-input positions
$X_i=p_i+az_i$, each unit $e\perp\mathbf1$ has

$$
\operatorname{Var}\left(e\cdot\frac1N\sum_iX_i\,
 \middle|\,\mathcal H_*\right)
=\frac{a^2q^2\chi}N>0.
\tag{NPG.13}
$$

Writing $D_{\rm col}$ for that color-orbit/terminal-geometry dictionary,
the same one-step executed update law, before discarding extinction, or
its actual survival-selected law gives

$$
\mathbb E\operatorname{Var}
\left(\sqrt N\,e\cdot N^{-1}\sum_iX_i\mid D_{\rm col}\right)
\ge a^2q^2\chi.
\tag{NPG.14}
$$

The statement retains every preceding companion, cloning and collision
correlation through its actual preparation. It excludes neither the
algorithm nor the color field: it identifies information genuinely omitted
by this particular same-record quotient. Adding the actual B2 position
array is a different existing descriptor refinement and removes this
specific missing centroid. Geometry recorded at the force-input stage
instead of the stated terminal geometry therefore does not inherit the
omission assertion.
:::

:::{prf:proof}
The native posterior decomposition is

$$
z_i=z_i^0+q\sqrt\chi\,
 (\widetilde\eta_i+N^{-1/2}Z_3),\qquad Z_3\sim N(0,I_3),
$$

where the relative residual array and $Z_3$ are independent conditional on
the full preparation and $Y$. The relative $r$ fixes only the relative
array. Resolve $Z_3$ into its normalized $\mathbf1$ direction and its two
orthogonal directions. Conditioning on $\mathbf1\cdot G$ fixes only the
first Gaussian coordinate; the other two remain independent standard
normals. This proves (NPG.12).

Changing $G$ by any $u\perp\mathbf1$ changes all B2 positions by $au$ and
all phase velocities by $u$. The Gaussian pair distances are unchanged,
so both normalized viscous forces, their norms and their validity masks
are unchanged. Every color is multiplied by
$U_u=\operatorname{diag}(e^{i\kappa u_1},e^{i\kappa u_2},e^{i\kappa u_3})$,
whose determinant is one. Its Gram coordinates are unchanged and its
complex determinants are unchanged. Triangle products and projector traces
are functions of those same invariant coordinates. The retained terminal
positions, their geometry and their support/selection masks are already
fixed. Hence every declared dictionary coordinate is $\mathcal H_*$-measurable.
This keeps the full complex determinant, rather than discarding its trace
phase or replacing it by its modulus.

The centroid of $X$ is $N^{-1}\sum p_i+aG$. Its projection onto either
trace-free unit direction is the fixed conditional mean plus a Gaussian
of variance $a^2q^2\chi/N$, proving (NPG.13). Because
$\sigma(D_{\rm col})\subset\mathcal H_*$, conditional variance decomposition
gives (NPG.14). Terminal survival in this one-step law depends only on $Y$,
already fixed in the conditioning, so the independent residual Gaussians
are unchanged by that actual selection. Every coefficient in the formulas
is an original algorithm/noise parameter.
:::

:::{prf:corollary} The hidden kinetic fiber persists under the actual stationary Doob weighting
:label: cor-npg-su3-doob-hidden-centroid

Keep the unchanged canonical instance whose primitive eigenfunction bound
is $e\ge m_N\ge\underline m_F>0$. Its full one-step stationary Doob
preparation/record law, relative to its QSD one-step surviving law, has
Radon--Nikodym derivative $e(S')/\nu(e)$ as in (NPT.12). Conditional on
$\mathcal H_*$ the exact trace-free density is therefore the original
standard Gaussian plane density multiplied by

$$
\frac{e(S'(Z_1,Z_2))}
 {\mathbb E[e(S'(Z_1,Z_2))\mid\mathcal H_*]}.
\tag{NPG.15}
$$

The cap and all velocity-dependent future features remain in $S'$; this
factor is not replaced by one. Its density is at least $m_N$ times the
original Gaussian density. Consequently

$$
\mathbb E_{\rm Doob}\operatorname{Var}
\left(\sqrt N\,e\cdot N^{-1}\sum_iX_i\mid D_{\rm col}\right)
\ge m_Na^2q^2\chi
\ge\underline m_Fa^2q^2\chi>0.
\tag{NPG.16}
$$

The physical-time calibration does not turn this conditional kinematic
variance into a physical gauge mass gap. At $q=0$ or $a=0$ this specific
centroid innovation is absent. At $s=0$ the terminal positions determine
that O-stage velocity and the terminal-geometry residual variance is zero.
Fixed-seed numerical execution has its different variance law. Future
survival through more than this update replaces (NPG.15) by its actual
remaining-survival tilt, which can bias the hidden plane; no Gaussian
independence under that future conditioning is asserted.
:::

:::{prf:proof}
The exact same-law identity (NPT.12) holds for the entire original latent
preparation and recorded update, because its density calculation cancels
the incoming eigenfunction and leaves only the terminal factor.
Conditioning that identity on $\mathcal H_*$ gives (NPG.15).
The numerator is at least $m_N$ and the denominator at most one. For every
constant $b$, its conditional second moment about $b$ is therefore at
least $m_N$ times the Gaussian conditional second moment about $b$.
Taking the infimum in $b$ proves the conditional variance bound. Conditional
variance decomposition onto $D_{\rm col}$ and averaging then give (NPG.16).
The stated degenerate cases follow directly from $a^2q^2\chi$ and the
original conditional Gaussian formula. A future-survival weight is the
actual conditional remaining survival function evaluated at $S'$;
it depends on the final cap velocity and is not constant on this fiber.
:::

(sec-npg-physical-sector-register)=
## 6. The actual physical-sector correspondence still required

:::{prf:proposition} Scope of the native gauge-sector tests
:label: prop-npg-physical-sector-register

The preceding results identify two existing gauge observation choices with
different information content. The primitive phase-space-ray projectors,
with their actual geometry coordinates, determine the full kinetic state
on regular eligible charts. The actual count-reference time-reversal cycle
is visible in their native clock-reflected algebra, with an explicitly
constructed native injection and a strictly positive transfer discrepancy.
Thus the literal positive-Hamiltonian identification on that complete
projector dictionary fails at fixed finite population; no invented CAR
mode or unobserved field was used to obtain this conclusion.

The direct common-$SU(3)$ orbit descriptor is a different native algebra.
Under the specified terminal-geometry conditioning it has two omitted
trace-free Gaussian centroid directions, with a derived conditional
variance lower bound. Its own reflected matrices have not been inferred
from the determining ray dictionary. The actual color-orbit hierarchy,
its history likelihoods and its refined predictive dynamics remain those
of {prf:ref}`thm-ym-native-physical-gauge-hierarchy` and
{prf:ref}`prop-sm-direct-markov-intertwining`. No instantaneous Markov
closure is supplied merely by its internal gauge invariance.

A uniform physical gap requires an actual determining physical dictionary,
its physical reflection/time implementer, a proved positive same-law transfer,
its nontrivial physical sector and the required uniform limit estimate.
The new finite-population decay and the new kinematic conditional variance
are not substitutes for those correspondences. Nor do the fixed-cutoff
projector defects prove that the corresponding physical determining defects
remain nonzero in every joint continuum scaling. That question needs their
actual cutoff/parameter estimates, on the intended physical observable class.
:::

(sec-npg-color-prediction-leakage)=
## 7. The actual color quotient does not close the native transition

:::{prf:theorem} Nonzero predictive leakage on the existing common-color orbit
:label: thm-npg-color-prediction-leakage

Use the unchanged canonical count reference with its actual B2 color
threshold $\delta_c=10^{-12}$, radial cap, terminal-position geometry,
stationary attached-record law $\widehat\pi$ and transition
$\widehat P$. More generally the same conclusion holds for the complete
count records satisfying the previously proved QSD, positive eigenfunction
and local-inverse certificates, with $q,s,a>0$, $E=1-a^2\lambda>0$ and
$B=a(1+c)>0$. Retain any original
$\delta_c\ge0$, including its exact zero-threshold branch. These are
primitive tests on the existing parameters, and the unchanged reference
satisfies them.

Let $D_{\rm col}$ retain the complete common-$SU(3)$ color orbit, its
original masks and terminal position array. It may also retain scalar
functions of the actual preceding preparation, and weights or geometry
functions invariant under a common translation of the B2 force-input
positions and velocities. Absolute B2 positions, the trace-free B2
centroid and individual component-frame projectors are not identified with
that quotient. No new condition is put on their actual stochastic law.

There is a bounded native terminal-position descriptor mode
$g\in L^2_0((D_{\rm col})_*\widehat\pi)$ for which

$$
\|L_{D_{\rm col}}g\|^2
=\mathbb E_{\widehat\pi}\operatorname{Var}
 \left(\widehat P(g\circ D_{\rm col})
       \mid D_{\rm col}\right)>0.
\tag{NPG.17}
$$

Thus this actual orbit dictionary does not define a strongly lumpable
current state for the existing update: its future predictor depends on
more than its present descriptor. This assertion does not alone rule out
a Markov law for a particular stationary observed process, or prove
$K_{D,n}\ne T_D^n$; those are different correlation statements. For every
candidate descriptor transfer $B_0$, without any self-adjointness requirement, its native injection has

$$
\|\widehat Cj_D-j_DB_0\|
\ge\|L_D\|>0.
\tag{NPG.18}
$$

This is an actual same-record omission. It does not conclude that the
common-$SU(3)$ descriptor process fails reflection positivity. A physical
reconstruction on its complete trajectory algebra can retain its memory.
Nor does (NPG.18) assert a population-uniform positive discrepancy.
The force-threshold masks are exactly invariant along the tested fiber,
so the conclusion retains every configured threshold. It does not require
an unavailable-color event.
:::

:::{prf:proof}
**1. A genuine native conditional fiber.** Keep the actual preparation
through A1 and denote it by $H$, with positions $p_i$ and first-kick
velocities $v_{1i}$. Conditional on $H$ and the terminal positions $Y$, the
original O-stage output has the proved Gaussian posterior. Write
$z_i=r_i+G$, where $\sum_i r_i=0$, and hold also
$\mathbf1\cdot G$ fixed. Conditional on this $\mathcal H_*$, the two
trace-free centroid coordinates have covariance $q^2\chi I_2/N$.
Under the stationary Doob law this posterior is multiplied by the
bounded positive terminal eigenfunction factor (NPG.15). Its conditional
density remains positive on every open set of that plane. No independence
under that tilt is assumed.

For every $u\perp\mathbf1$, replacing $G$ by $G+u$ translates all B2
positions by $au$ and velocities by $u$. It leaves the actual viscous
forces and all force-threshold masks unchanged. The color vectors acquire
the same diagonal $SU(3)$ factor, so their complete Gram/determinant orbit
and projector traces are exactly unchanged, as proved in
{prf:ref}`thm-npg-su3-hidden-centroid`. The terminal geometry is already
fixed by $Y$. Every declared translation-invariant weight and preparation
scalar is fixed on this fiber. Thus the whole declared dictionary is
constant as the two trace-free coordinates of $G$ vary, for every actual
force threshold. The statement uses the full complex determinant, whose
trace phase is retained by fixing $\mathbf1\cdot G$.

The actual uncapped second-kick velocity on this fiber is

$$
w_i=EG-a\lambda\overline p+\widetilde w_i(H,r),
\qquad \sum_i\widetilde w_i=0,
\tag{NPG.19}
$$

where the relative term has no $G$ dependence. This follows from the
symmetry of the count kernel and its zero total viscous force. The final
core velocity is exactly $C_V(w_i)$. All those changes are changes of an
original O innovation on the same recorded fiber.

The stationary preparation has positive probability arbitrarily close to
$p=v_1=0$: the proved common part gives positive incoming QSD mass near
all-alive zero coordinates; a no-accepted-clone pattern has positive
probability there; its actual first-kick and A1 maps are continuous and
send zero to zero. Conditional Gaussian smoothing gives positive density
of $Y,r$ near zero for those preparations. The scalar centroid trace also
has a nondegenerate conditional Gaussian density before its bounded Doob tilt. Hence the conditioning parameters
$(p,v_1,Y,r,\mathbf1\cdot G)$ approach zero through neighborhoods of
positive stationary attached probability. The other components of $H$
retain their original values; “zero preparation” concerns its continuous
position/velocity coordinates. In particular the proof does not
condition on an exactly consensus event of probability zero.

**2. What zero leakage would force.** Suppose every bounded
terminal-position test has zero leakage. Use a countable determining
family of indicators of rational position boxes. Its conditional predictor
would then be $D_{\rm col}$-measurable. Conditioning further on
$\mathcal H_*$ would make its value constant for almost every trace-free centroid coordinate, because the dictionary is
constant on that genuine conditional fiber.

The future core kernel is continuous in total variation in the entering
coordinates on the all-alive neighborhood in question. To check the
continuity actually used here, freeze the finitely many companion,
donor and accepted-edge patterns; their positive standardized fitness and
Gaussian donor weights are continuous there. Each conditional collision
and kinetic map varies continuously in its entering coordinates. Its
almost-everywhere nonsingular Gaussian pushforward has the total-variation
continuity of {prf:ref}`thm-cgd-phase-smoothing`. Integration over the
original rotations and jitters preserves that continuity by domination.
At an all-alive consensus every accepted-gate probability tends to zero,
so any additional accepted pattern has vanishing total mass. Thus the same
continuity holds at consensus, even if a discrete tie rule has several
representations there. Terminal absorption contracts total variation.
Finally $e=\alpha_N^{-1}Qe$ and $m_N>0$ give continuity of $e$ at those
points and hence of the actual Doob rows. No independent regularity
hypothesis on $e$ is added.

Therefore the predictor is constant for every trace-free centroid point
on each of the almost-everywhere fibers under consideration. Choose such
conditioning values with $(p,v_1,Y,r,\mathbf1\cdot G)\to0$. Fix a unit
physical-component vector $\ell\perp\mathbf1$. Equation (NPG.19), the actual
cap and the just proved continuity would imply that the next
terminal-position law from

$$
S_u:\quad x_i=0,\qquad v_i=u\ell
$$

is the same for all $u$ in any fixed compact interval inside $(0,V)$.
Overlapping intervals give one common marginal on all of $(0,V)$.
No limit of a normalized color direction is used: the orbit value may
vary with the conditioning sequence, while its predictor is constant along
each actual centroid fiber. The next calculation rules out that consequence
for the actual Doob transition.

**3. Exact count-consensus exponential family.** From $S_u$ all original
fitnesses agree, so no clone is accepted and each component collision is
the identity. The first kick has zero force. With the original OU and
position innovations, the next positions and uncapped velocities satisfy

$$
Y_i'=Bu\ell+aq\xi_i+s\zeta_i,
\qquad
w_i'=Du\ell+Eq\xi_i-a\nu L_{aq\xi}(q\xi)_i,
\quad D=c-a^2\lambda(1+c).
$$

The Gaussian count Laplacian is symmetric. Consequently its mean is
zero, and its dependence on the OU array is only through relative rows.
The normalized centroid pair
$M=\sqrt N\,\overline Y'$,
$W=\sqrt N\,\overline w'$ is therefore Gaussian with mean
$\sqrt N(Bu,Du)\ell$ and component covariance

$$
\Sigma=\begin{pmatrix}a^2q^2+s^2&aq^2E\\
aq^2E&q^2E^2\end{pmatrix}.
\tag{NPG.20}
$$

It is independent of the relative output pair, whose entire nonlinear
coupled law is unchanged as $u$ varies. Passing through the original cap
is bijective and does not change the likelihood ratio. Killing depends on
the terminal position array. Thus the actual killed kernels have the
exact likelihood relation

$$
\frac{dQ(S_u,\cdot)}{dQ(S_0,\cdot)}
=\exp\{u\mathcal T-\tfrac12N\Lambda u^2\},
\quad
\Lambda=(B,D)\Sigma^{-1}(B,D)^{\mathsf T},
\tag{NPG.21}
$$

where $W$ is recovered from the actual final velocities by the cap inverse,
and

$$
\mathcal T=\sqrt N\left\{\frac a{Es^2}\ell\cdot M
                 +\kappa_{\rm cap}\ell\cdot W\right\}.
\tag{NPG.22}
$$

Indeed $(B,D)\Sigma^{-1}=(a/(Es^2),\kappa_{\rm cap})$ by direct
multiplication. All relative coupled errors, terminal masks and cap
Jacobians cancel in (NPG.21), rather than being discarded.

Let $P_0=P(S_0,\cdot)$. Multiplication by the actual terminal
eigenfunction gives

$$
P(S_u,ds')=
 \frac{e^{u\mathcal T(s')}}{\mathbb E_{P_0}e^{u\mathcal T}}P_0(ds').
\tag{NPG.23}
$$

Normalization is the eigenfunction equation at $S_u$. The bounded
positive tilt of the original Gaussian centroid makes every real
exponential moment finite.

**4. The geometry cannot lose this whole family.** If the $Y'$ marginal
of (NPG.23) were identical for all $u\in(0,V)$, the proved total-variation
continuity at $u=0$ would identify its common marginal with that of $P_0$.
Its conditional moment generating function would then satisfy

$$
\mathbb E_{P_0}[e^{u\mathcal T}\mid Y'=y]
 =\mathbb E_{P_0}e^{u\mathcal T}
\tag{NPG.24}
$$

for almost every $y$. First use a countable dense set of $u$ and then
continuity to retain one common full-measure set. Gaussian exponential
moments make both sides analytic, so equality on an interval determines
the conditional distribution of $\mathcal T$. It would be independent
of $Y'$.

If $\kappa_{\rm cap}=0$, (NPG.22) makes $\mathcal T$ a nonconstant
function of $Y'$ on its all-alive full support. It cannot be independent of
$Y'$, already contradicting (NPG.24). This disposes of that exact parameter
regime without changing its position-noise or cap parameters.

For $\kappa_{\rm cap}\ne0$, before its bounded terminal eigenfunction
tilt, conditioning the complete future position array $Y'=y$ makes
$\mathcal T$ Gaussian with mean and variance

$$
\mu_y=\frac{NB}{a^2q^2+s^2}\,\ell\cdot\overline y,
\qquad
\sigma_{\mathcal T}^2
 =N\kappa_{\rm cap}^2q^2E^2\chi>0.
\tag{NPG.25}
$$

Independence of the centroid and relative Gaussian inputs justifies this
conditioning on the entire array. Integrating out relative velocities and
the other centroid components multiplies this one-dimensional Gaussian
density by a factor between $m_N$ and one. After conditional normalization,
its actual $P_0$ density $h_y$ therefore obeys

$$
m_N\varphi_{\mu_y,\sigma_{\mathcal T}^2}
\le h_y\le m_N^{-1}\varphi_{\mu_y,\sigma_{\mathcal T}^2}.
\tag{NPG.26}
$$

There are two all-alive position arrays in the actual support with distinct
centroids in the $\ell$ direction, and such arrays can be selected from
the full-measure set in (NPG.24). If the conditional law were the same at
both, (NPG.26) would bound the ratio of their two Gaussian densities
between $m_N^2$ and $m_N^{-2}$ on the whole line. Distinct means with the
same positive variance have an exponential density ratio unbounded in
one tail. This is a contradiction.

Some native bounded terminal-position test thus has a predictor that is
not constant on the actual trace-free conditional fibers. Conditional
variance is zero exactly for descriptor-measurable predictors. Subtracting
the stationary mean from this test therefore gives the mode in (NPG.17).
The leakage identity of (NPG.10) gives (NPG.18) for every candidate
operator. Nothing in this proof replaces the native transition by a
freely chosen exponential, conditions away a rare adverse outcome or
assigns an artificial local mode.
:::
