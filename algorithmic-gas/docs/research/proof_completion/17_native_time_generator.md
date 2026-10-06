# Native algorithmic-time generator, jump response and scaling regimes

(sec-ntg-register)=
## 1. The complete execution record and actual parameter families

:::{prf:definition} Native time-scaling register
:label: def-ntg-complete-register

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`, the canonical donor,
fitness and component-collision convention of
{prf:ref}`def-cgd-parameter-register`, and the independently sampled
real-coordinate Gaussian and Haar laws. Let $N,d$ be fixed, with $N\ge2$;
retain every companion bandwidth, feature clamp, distance regularizer,
global standardization scale, objective orientation, positive-map amplitude
and positive floor, fitness exponent, current-donor self-exclusion rule,
clone jitter $\sigma_J\ge0$ and component restitution
$\alpha_{\rm col}\in[0,1]$. The two companion roles are the canonical
independent sampled roles. The cloning period is one, with no elite
restoration or historical donor pool. All rows are initially eligible.

The positive generator regime uses the EXISTING B-A-O-A-B branch with
quadratic force $F_{\rm pot}(x)=-\lambda x$, $\lambda\ge0$, either count
or row Gaussian viscosity $\nu\ge0$, $\rho>0$, friction $\gamma\ge0$,
OU amplitude $b_O\ge0$, independent position diffusion $\sigma_x\ge0$,
and the configured B2/final recording stage. The existing boundary tag is
`Unbounded`; consequently no terminal death or mandatory revival occurs
in this regime. Recording is passive: no metric, reward, Boris, curl or
noise feedback is consumed beyond the named reward/force. The raw reward
is the configured quadratic reward, with its actual orientation. The
global logistic positive maps have amplitudes $A_r,A_s>0$, floors
$\eta_r,\eta_s>0$, and exponents $p_r,p_s\ge0$.

In a family indexed by the ACTUAL timestep $h\downarrow0$, keep these
parameters fixed and set only the existing acceptance saturation to
$s_c(h)=s_0/h$, $s_0>0$. The actual acceptance regularizer is fixed
$\epsilon_c\ge0$. The velocity cap is either `None` or its existing
radius $V(h)>0$ satisfying
$1/V(h)=\kappa h+o(h)$, $\kappa\ge0$. Thus $\kappa=0$ includes the
uncapped branch and radii growing faster than $1/h$. This is a specified
family of existing configurations, not the unchanged $h=.04,V=2,s_c=1$
reference.

Let $S=(x,v)\in\mathbb R^{2Nd}$ denote its complete swarm coordinates.
Its passive histories retain their original functions of these coordinates
and update draws. The generator below concerns the coordinate state,
rather than declaring a reduced color descriptor to be Markovian.
Put
$$
f_-=\eta_r^{p_r}\eta_s^{p_s},\qquad
f_+=(A_r+\eta_r)^{p_r}(A_s+\eta_s)^{p_s},\qquad
R_*=\frac{f_+-f_-}{s_0(f_-+\epsilon_c)}.
\tag{NTG.1}
$$
A zero exponent contributes one. These are primitive global fitness and
rate bounds; no property of an unknown stationary law is assumed.
:::

:::{prf:definition} Actual one-edge jump and sampled rates
:label: def-ntg-native-one-edge-jump

Let $\omega$ be the complete pair of sampled companion arrays before
cloning and let $j_i(\omega)$ be row $i$'s actual current clone donor.
Write $q_S(\omega)$ for its actual joint sampling probability, and
$f_i(S,\omega)$ for the fitness after the actual sampled diversity and
global standardizations. Define
$$
r_i(S,\omega)=
\frac{[f_{j_i(\omega)}(S,\omega)-f_i(S,\omega)]_+}
 {s_0[f_i(S,\omega)+\epsilon_c]}\in[0,R_*].
\tag{NTG.2}
$$
Singleton self-donors have rate zero.

For $i\ne j$, $T_{ij}^{J,O}S$ is the existing simultaneous clone transform
when its accepted graph has exactly the one edge $i\leftarrow j$.
It replaces $x_i$ by $x_j+\sigma_JJ$ with $J\sim N(0,I_d)$, leaves
other positions fixed, and applies the configured shared component
rotation to the frozen pre-copy velocities on $\{i,j\}$:
$$
m_{ij}=(v_i+v_j)/2,\qquad
v_i'=m_{ij}+\alpha_{\rm col}O(v_i-m_{ij}),\quad
v_j'=m_{ij}+\alpha_{\rm col}O(v_j-m_{ij}).
\tag{NTG.3}
$$
The other velocities are unchanged, and $O$ has the actual Haar law.
If collision restitution is disabled in the existing transform, replace
(NTG.3) by its literal source copy $v_i'=v_j$, with other velocities fixed.
This optional branch retains its tag in $\mathfrak P$. No independent
recipient rotation replaces the component rotation.

Let $\mathcal K_{ij}\phi(S)=E_{J,O}\phi(T_{ij}^{J,O}S)$. All jitters
remain unbounded. Finally let the actual force on a state be
$$
F_i^{\mathfrak n}(S)=-\lambda x_i+
\nu\sum_{j\ne i}a_{ij}^{\mathfrak n}(x)(v_j-v_i),
$$
where $a_{ij}^{\rm count}=N^{-1}e^{-|x_i-x_j|^2/(2\rho^2)}$ and
$a_{ij}^{\rm row}$ is the existing same-row normalized Gaussian weight.
:::

(sec-ntg-generator)=
## 2. The complete native drift and bracket in this evaluated regime

:::{prf:theorem} Actual complete-step generator including selection and shared collisions
:label: thm-ntg-complete-generator

Let $P_h$ be the actual complete-step conservative kernel in
{prf:ref}`def-ntg-complete-register`. For every
$\phi\in C_b^3(\mathbb R^{2Nd})$ with bounded derivatives through order
three,
$$
\frac{P_h\phi-\phi}{h}\longrightarrow\mathcal A\phi
\quad\hbox{locally uniformly in }S,
\tag{NTG.4}
$$
where
$$
\begin{aligned}
\mathcal A\phi={}&
\sum_i\left[
 v_i\cdot\nabla_{x_i}\phi+
 (F_i^{\mathfrak n}-\gamma v_i-\kappa|v_i|v_i)
                      \cdot\nabla_{v_i}\phi\right]\\
&+\frac{\sigma_x^2}{2}\sum_i\Delta_{x_i}\phi
 +\frac{b_O^2}{2}\sum_i\Delta_{v_i}\phi\\
&+\sum_\omega q_S(\omega)
   \sum_i r_i(S,\omega)
       [\mathcal K_{i,j_i(\omega)}\phi(S)-\phi(S)] .
\end{aligned}
\tag{NTG.5}
$$
In particular the actual sampled global fitness and source choices remain
inside their joint expectation; replacing them by the fitness of an
averaged companion is not this generator.

For bounded $C^3$ tests $\phi,\psi$, the native complete-step conditional
covariance satisfies, locally uniformly,
$$
\frac{\operatorname{Cov}_S(\phi(S_1),\psi(S_1))}{h}
\longrightarrow\mathcal B(\phi,\psi)(S),
\tag{NTG.6}
$$
with the full bracket
$$
\begin{aligned}
\mathcal B(\phi,\psi)={}&
 \sigma_x^2\sum_i\nabla_{x_i}\phi\cdot\nabla_{x_i}\psi+
 b_O^2\sum_i\nabla_{v_i}\phi\cdot\nabla_{v_i}\psi\\
&+\sum_\omega q_S(\omega)\sum_i r_i(S,\omega)
 E_{J,O}\big[
   (\phi(T_{ij}^{J,O}S)-\phi(S))
   (\psi(T_{ij}^{J,O}S)-\psi(S))\big],
\quad j=j_i(\omega).
\end{aligned}
\tag{NTG.7}
$$
These are proved infinitesimal identities for the full native coordinates.
They do not alone prove a field limit, its phase selection or a
Yang--Mills action.
:::

:::{prf:proof}
The logistic positive maps give $f_-\le f_i\le f_+$ after every actual
standardization. Conditional on $\omega$, the accepted-edge indicators
are the actual independent Bernoulli draws with probability
$\min(1,h r_i)$. For $hR_*<1$ this is exactly $h r_i$, so the chance
of at least two accepted edges is at most
$N(N-1)h^2R_*^2/2$. The empty graph has probability
$1-h\sum_i r_i+O(h^2)$, uniformly in $S,\omega$, and the probability
of its single edge $i\leftarrow j_i$ is $hr_i+O(h^2)$.
The finite companion sum is over the original sampling law.
On the empty graph the transform is the identity, and on the one-edge
graph it is exactly (NTG.3), because the code computes component centers
and relative velocities from the frozen pre-copy array. Bounded tests
make all multiple-edge terms $O(h^2)$ without conditioning away any draw.

On the empty graph, B1 gives $v_1=v+hF(S)/2$. The first A update is
$x_1=x+hv_1/2$. The actual O coefficient and variance satisfy
$$
e^{-\gamma h}=1-\gamma h+O(h^2),\qquad
q_h^2=b_O^2h+O(h^2).
$$
Then $z=e^{-\gamma h}v_1+q_h\xi$, the second A update is
$X=x_1+hz/2$, and B2 gives $w=z+hF(X,z)/2$.
Only after B2, the actual independent final position diffusion gives
$Y=X+\sigma_x\sqrt h\,\zeta$. The $\xi_i,\zeta_i$ are the original independent
normal rows. Thus, on any compact set of entering states,
$$
Y-x=hv+\sigma_x\sqrt h\,\zeta+o_{L^p}(h),
\qquad
w-v=h(F(S)-\gamma v)+q_h\xi+o_{L^p}(h)
\tag{NTG.8}
$$
for the drift expansion, with the first error understood after subtracting
the centered $O(h^{3/2})$ A-stage velocity noise.
For the second expression its force-difference remainder is
$O_{L^p}(h^{3/2})$.

Here is the needed control at unbounded innovations. For both actual
normalizations, $\sum_{j\ne i}a_{ij}\le1$; consequently
$|F_i|\le\lambda|x_i|+2\nu\max_j|v_j|$.
The count Gaussian force is smooth. For the row branch its denominator
is positive at every finite state; on a fixed compact neighborhood its
derivatives are bounded by the explicitly differentiated finite
Gaussian sums. Restrict only the estimate to
$|\xi|+|\zeta|\le h^{-1/8}$, where the stage arguments stay in that
neighborhood. The complementary original Gaussian event has
probability at most $C e^{-c h^{-1/4}}$; its polynomial moments obey the
same exponential bound after weakening $c$. The global linear force
bound above controls the stage variables there. This proves the local
$L^p$ remainder statements for every fixed $p$, without clipping the law.

If a cap is present, its exact identity is
$$
C_{V(h)}(w)-w
=-\frac{V(h)^{-1}|w|w}{1+V(h)^{-1}|w|}
=-\kappa h|v|v+o_{L^p}(h)
$$
on those compact input sets. Its remainder is bounded using the same
normal moments. The leading conditional covariance of $(Y,w)$ is
$h\operatorname{diag}(\sigma_x^2I,b_O^2I)$; the covariance between
its two coordinate blocks is $O(h^2)$, since the O contribution to
position has the additional A coefficient $h/2$. A third-order Taylor
expansion with bounded derivatives now gives the differential part of
(NTG.5), with a remainder $O(h^{3/2})+o(h)$.

On a single accepted edge, its post-transform coordinates have every
fixed moment uniformly for entering $S$ on a compact set, because
$J$ is the original Gaussian and the component map is linear in
the bounded entering velocities. The same linear force bound implies
that the subsequent kinetic increment tends to zero in $L^p$.
The cap tends to the identity. Hence its expected bounded Lipschitz
test differs from $\mathcal K_{ij}\phi$ by $o(1)$ uniformly there.
Multiplication by its probability $O(h)$ gives $o(h)$; no
differentiability of the positive-part acceptance at a fitness tie is
needed. This proves (NTG.4)--(NTG.5).

Apply (NTG.4) also to $\phi\psi$, whose derivatives remain bounded.
Since $P_h\phi=\phi+h\mathcal A\phi+o(h)$,
$$
h^{-1}\operatorname{Cov}_S(\phi(S_1),\psi(S_1))
=\mathcal A(\phi\psi)-\phi\mathcal A\psi-\psi\mathcal A\phi+o(1).
$$
The first-order drift terms cancel. The diffusion product rule gives
the two gradient terms in (NTG.7); expansion of the actual jump
product gives its last expectation. This proves the full bracket.
:::

:::{prf:corollary} Actual smooth gauge tests inherit this drift and bracket
:label: cor-ntg-native-gauge-tests

In the positive generator regime, any bounded smooth test of an existing
native state-derived gauge descriptor on a chart with its actual
normalization denominators separated from zero has drift and bracket
(NTG.5)--(NTG.7), applied to its actual coordinate pullback, after
multiplication by a smooth chart cutoff. In particular all source,
selection and collision terms remain. The joint law of the descriptor
need not be Markovian. A B1/B2 descriptor using newly drawn or
transition-attached variables must instead retain that observation
kernel; it is not replaced by a state-derived smooth field.
:::

:::{prf:proof}
The chart and cutoff make the actual coordinate pullback a bounded
$C^3$ test with bounded derivatives. Apply the theorem and its product
calculation. Conditioning the resulting increments on a coarser
descriptor gives its true conditional drift and covariance; it does
not remove their dependence on unresolved state coordinates.
:::

:::{prf:theorem} Full finite-population native process in the positive scaling regime
:label: thm-ntg-full-process-limit

In {prf:ref}`def-ntg-complete-register`, let the initial coordinate
state be fixed, or have its specified law with finite second moment.
The actual paths $S_h(t)=S_{\lfloor t/h\rfloor}$ converge in law on
every finite interval in the cadlag Skorokhod topology to the uniquely
constructed nonexplosive finite-population jump diffusion with generator
(NTG.5). Between jumps it solves
$$
dX_i=V_i\,dt+\sigma_x\,dB_i^x,\qquad
dV_i=(F_i^{\mathfrak n}(X,V)-\gamma V_i-\kappa|V_i|V_i)\,dt
                                      +b_O\,dB_i^v.
\tag{NTG.9}
$$
Its jump intensities, companion/fitness sampling, Gaussian recipient
jitter and shared two-member Haar transforms are exactly
(NTG.2)--(NTG.3). This gives a full native coordinate drift/bracket limit
for this specified family, without assuming stationary chaos or
a field-level functional inequality.
:::

:::{prf:proof}
All actual finite companion probabilities are locally Lipschitz
functions of the coordinates: feature clamps are Lipschitz, the
positive Gaussian normalizers have a positive minimum on each compact
set, the distance regularizer is positive, and the global variance
regularizers are positive. The sampled quadratic reward, logistic
maps and positive-part rates are locally Lipschitz there as well.
The force is locally Lipschitz for both normalizations. Its global
linear growth estimate was proved above. The vector $|v|v$ is
continuously differentiable and locally Lipschitz.

Construct candidate jump times from a Poisson process of rate $NR_*$
(if $R_*=0$, omit it). At each candidate choose $i$ uniformly, draw
the original $\omega$ from $q_S$, and accept with the original rate
threshold $r_i/R_*$. Apply $T_{i,j_i(\omega)}^{J,O}$ on acceptance.
This exactly has the jump part of (NTG.5). There are almost surely
finitely many candidates on each finite interval. Between them,
subtract the additive Brownian path from (NTG.9) and solve its locally
Lipschitz integral equation by successive Picard iterations up to
exit from each coordinate ball. The usual factorial bound on
successive iterates on a fixed sufficiently short bounded interval
proves existence and uniqueness there. Iterate until exit or the
next candidate. This constructs a unique path from the Brownian
motions and these independent candidate marks up to possible explosion.

Set $E(S)=1+\sum_i(|x_i|^2+|v_i|^2)$. The cap drift contributes
$-2\kappa\sum_i|v_i|^3\le0$. For both normalized forces the linear
growth bound gives
$\mathcal A_{\rm diff}E\le C_NE+C_N$ for an explicit finite
$C_N$ depending only on $N,d,\lambda,\nu,\gamma,\sigma_x,b_O$.
For example one may use $C_N$ enlarged from
$2(1+\lambda)+4\nu N+Nd(\sigma_x^2+b_O^2)+1$.
The one-edge position replacement has mean increment at most
$\sum_i|x_i|^2+d\sigma_J^2$.
When the configured collision is enabled, the pair kinetic energy
is exactly
$$
2|m_{ij}|^2+\frac{\alpha_{\rm col}^2}{2}|v_i-v_j|^2
\le |v_i|^2+|v_j|^2.
$$
When collision is disabled, its copy increases the full velocity
energy by at most $\sum_i|v_i|^2$.
Consequently the total jump contribution obeys
$\mathcal A_{\rm jump}E\le NR_*(E+d\sigma_J^2)$.
Stopped Ito expansion, including the compensated finite Poisson sum,
and the integral Gronwall inequality give
$E[E(S_{t\wedge\tau_K})]\le(E[E(S_0)]+Ct)e^{Ct}$ for a finite
primitive $C$, where $\tau_K$ is first exit from the energy ball
$E<K$. An overshooting jump has finite mean by the Gaussian second
moment, so the same stopped identity applies.
On $\{\tau_K\le T\}$ the stopped energy is at least $K$.
The resulting exit probability is at most
$(E[E(S_0)]+CT)e^{CT}/K$. Letting $K\to\infty$ proves nonexplosion.

The discrete chains have the same compact-containment bound uniformly
for small $h$. Conditional on the companion array their accepted
probabilities are at most $hR_*$. Source copying and the Gaussian
jitter give
$$
E\!\left[\sum_i|x_i^c|^2\mid S\right]
\le(1+NhR_*)\sum_i|x_i|^2+Nd\,hR_*\sigma_J^2.
$$
All actual component collisions contract the frozen total velocity
energy when enabled; if disabled, the analogous copy bound is
$(1+NhR_*)\sum_i|v_i|^2$.
The quadratic kinetic stages and their Gaussian moments give
$E[E(S_1)\mid S]\le(1+Ch)E(S)+Ch$ for small $h$.
The actual cap only decreases energy. Applying this inequality up
to first exit, and the discrete Gronwall inequality, proves the
same exit bound with a harmless enlarged $C$.

It remains to identify convergence, rather than infer it only from
generator convergence. Replace the actual Bernoulli accepted plan
at each step, for the purpose of coupling, by one candidate of
probability $NR_*h$, uniform row $i$, original companion sample,
and threshold $r_i/R_*$. The probabilities of its empty and
single accepted plans differ from the actual plan by $O(h^2)$,
uniformly in the entering state; all multiple-edge plans have
probability $O(h^2)$. The following jitter and Haar laws are
identical on agreeing plans. A maximal coupling therefore changes
no actual coordinate path over $T/h$ steps with probability
$1-O(T h)$. Replace that candidate schedule by the Poisson schedule:
its probability of two candidates in a step and the difference of
its one-candidate probability from $NR_*h$ are also $O(h^2)$.
The total mismatch probability again tends to zero.
Thus these replacements couple laws; they are not modifications
of the defined algorithm.

On an energy ball, couple the actual OU innovation to
$b_O\int_0^h e^{-\gamma(h-u)}\,dB^v_u$ and the actual final
position innovation to $\sigma_x(B^x_h-B^x_0)$ on each interval.
The former differs from $b_O(B^v_h-B^v_0)$ in second mean by
$O(h^3)$, by the Ito isometry. The bounded local force
Lipschitz constants, the cap identity and the two A stages give
a local drift remainder with first mean $O(h^{3/2})+o(h)$.
The ordinary additive-noise Euler comparison on that ball now
converges uniformly in probability between candidate jumps:
subtract the two coupled integral equations, use the local
Lipschitz bound, sum these remainder expectations, and apply
the integral Gronwall inequality. Brownian interpolation errors
vanish by its continuity. The centered OU-error sums have squared
maximal expectation $O(h^2)$ by the elementary martingale maximal
inequality and the preceding isometry.

At the finitely many Poisson candidates, $q_S$ and $r_i$ are
continuous. Draw their finite companion outcomes with the same
uniform variables in an inverse-cumulative construction.
Almost surely none of those uniforms lies on a limiting
cumulative-probability boundary, or on its acceptance threshold.
Thus agreeing approaching states give agreeing outcomes for
sufficiently small $h$, almost surely, at each candidate.
The same $J,O$ then give converging post-jump coordinates, since
the actual maps (NTG.3) are continuous in their entering state.
Induct over the finitely many candidates. Their times can be
moved to the discrete commit times with a Skorokhod time change
of maximum error $h$; candidates within $h$ of the final endpoint
have probability $O(h)$ and can be excluded from the estimate.
This proves stopped path convergence. The two compact-containment
bounds remove the stops, and truncation of the specified initial
law handles its finite second moment. The constructed limiting
law has exactly (NTG.9) and (NTG.5)--(NTG.7).
:::

(sec-ntg-native-projector-process)=
## 3. The actual B2 gauge process and its nonzero full bracket

:::{prf:theorem} Native B2 projector dynamics on its actual valid charts
:label: thm-ntg-native-projector-process

Retain the positive scaled family of
{prf:ref}`thm-ntg-full-process-limit` and its existing matched B2
color instrument, with $d=3$, phase scale $\kappa_c\ne0$ and finite
force threshold $\delta_c\ge0$. The cap coefficient $\kappa$ in
(NTG.5) is distinct from this color phase scale.
Define the actual state color and projector on its available branch by
$$
c_i(S)=
 \frac{F_i^{\rm visc}(S)}{|F_i^{\rm visc}(S)|}
                  \odot e^{i\kappa_c v_i},\qquad
P_i(S)=c_i(S)c_i(S)^\dagger.
\tag{NTG.10}
$$
Let $\Phi$ be any bounded $C^3$ cylinder of finitely many such projectors,
with a smooth coordinate cutoff on a bounded chart separated from the
actual force thresholds and other consumed readout denominators.
Extend its pullback by zero outside that chart.

The existing B2 record process for this cylinder differs in maximum
norm over a fixed finite native-time interval from $\Phi(S_h)$ by
a quantity tending to zero in probability. Consequently it converges
to $\Phi(S)$, where $S$ is the actual limiting jump diffusion just
constructed. Its full drift is $\mathcal A\Phi$ and its full
quadratic/mixed bracket is $\mathcal B(\Phi,\Psi)$, with every
selection, donor, shared collision and jitter term in
(NTG.5)--(NTG.7). The cutoff has its own derivatives in these expressions;
it does not assert an unconditioned limit for a hard availability mask.

There is an explicit nonzero four-dimensional native diffusion regime.
Take $N\ge2$, $\nu,b_O>0$, $R>\delta_c$, and the allowed entering
state $x_i=0$ for all rows,
$$
v_1=0,\qquad v_2=\frac{n_*R}{\nu}n,\qquad
v_i=0\ (i>2),\qquad
n=(1,1,1)/\sqrt3,\qquad
n_*=\begin{cases}N,&{\rm count},\\N-1,&{\rm row}.\end{cases}
\tag{NTG.11}
$$
At its tag-$1$ projector $P_0=nn^\dagger$, use the real four-coordinate
chart $\zeta(P)=Q_0Pn$ with $Q_0=I-P_0$, identified with
$\mathbb C^2$ in any real orthonormal pair in $n^\perp$.
The continuous part of its actual bracket obeys
$$
\mathcal B_{\rm diff}(\zeta,\zeta)
\ \ge\ \frac{b_O^2}{D_*}I_4,\qquad
D_*=\left(\frac{n_*R}{\nu}\right)^2+
                \frac{3[1+(N-1)^2]}{\kappa_c^2}.
\tag{NTG.12}
$$
The full bracket adds its nonnegative jump covariance.
On a smaller valid neighborhood the bound remains at least
$b_O^2/(2D_*)$. Its native projector connection has noncommuting
curvature there, as in
{prf:ref}`thm-npc-native-curvature-nondegeneracy`.
Thus this is a nonzero complete native gauge dynamics regime,
not merely a finite nonzero color amplitude. It is a fixed-population
continuum-time conclusion for this parameter family; it is not the
population/graph continuum Yang--Mills identification.
:::

:::{prf:proof}
On a bounded valid chart the actual force and color maps have bounded
derivatives of every fixed order. At the completed step, B2 uses its
pre-position-diffusion coordinates $X$ and its uncapped O velocities
$z$. The final coordinates satisfy
$Y-X=\sigma_x\sqrt h\,\zeta$ and
$v^+-z=hF(X,z)/2-\kappa h|z|z+o(h)$ on bounded stage sets.
For at most $T/h$ independent Gaussian rows, their maximum norm is
$O_P(\sqrt{\log(1/h)})$: the elementary Gaussian tail and union
bound give this statement for any fixed $N,T$.
The maximum difference between the B2 arguments and the final
coordinates therefore tends to zero in probability.
The finite-rate accepted jumps retain all their exact post-copy
positions and velocities in BOTH records; they do not add a
difference between these two stage arguments.
Uniform chain compact containment and the finite number of limiting
candidate jumps remove the bounded-stage restriction; the finite
Gaussian clone marks have tight maxima on that window.
The cylinder's smooth extension and cutoff then give the claimed
maximum difference. Apply the full path convergence and continuity
of this coordinate cylinder, followed by the generator/product
calculation, to obtain its complete drift and bracket.

At (NTG.11), the actual kernels equal one and their first position
derivatives vanish. The tag force is $Rn$ and its differential is
$$
\delta F_1^{\rm visc}
=\frac{\nu}{n_*}
 \left[\sum_{j\ne1}\delta v_j-(N-1)\delta v_1\right].
$$
The row denominator derivative is also zero at this same-position
state. Its horizontal color differential is
$$
w=\frac1R Q_0\delta F_1^{\rm visc}
       +i\kappa_c Q_0\operatorname{diag}(n)\delta v_1 .
$$
The projector chart differential is exactly $w$, since
$\delta P=wn^\dagger+nw^\dagger$ and $Q_0\delta Pn=w$.
For any $u,v\in n^\perp$ real, obtain $w=u+iv$ by the actual
velocity changes
$$
\delta v_1=\frac{\sqrt3}{\kappa_c}v,\qquad
\delta v_2=\frac{n_*R}{\nu}u+
                    (N-1)\frac{\sqrt3}{\kappa_c}v,\qquad
\delta v_j=0\ (j>2).
$$
Their squared norm is at most $D_*(|u|^2+|v|^2)$, by the
Cauchy--Schwarz inequality for the two coefficients in $\delta v_2$.
This constructs a right inverse $T$ of the real chart Jacobian $D$
with $\|T\|^2\le D_*$. For any real chart covector $a$,
$|a|^2=\langle D^\mathsf Ta,Ta\rangle$, so
$|D^\mathsf Ta|\ge |a|/\sqrt{D_*}$.
The original velocity diffusion is $b_O$ times independent real
Brownian rows. Its chart covariance is $b_O^2DD^\mathsf T$,
proving (NTG.12). The position covariance and jump bracket are
nonnegative and cannot remove this bound.

The actual force threshold is strictly passed. Choose a chart cutoff
equal to one near this point. Continuity preserves the smaller
positive covariance bound on a neighborhood.
The three tangent pairs in the cited native curvature proof use
precisely the real directions $e,f,if$ in this differential range;
their two complement curvature matrices have nonzero commutator.
This gives the same non-Abelian execution-chart curvature.
Starting the existing scaled process at (NTG.11), its continuous
paths before its first candidate stay in that neighborhood for
a strictly positive random time almost surely. The candidate clock
also has strictly positive first waiting time almost surely.
Hence its nonzero diffusion dynamics is an actual positive-time
property of this specified initial regime, with no stationary
support assumption. All constants in the covariance are evaluated
from the primitive finite population, viscosity, phase and noise
parameters.
:::

(sec-ntg-literal-reference)=
## 4. Characterization of fixed-parameter small-step families

:::{prf:theorem} Fixed native cap or active acceptance need not be infinitesimal
:label: thm-ntg-fixed-reference-noninfinitesimal

Retain the actual terminal-box or unbounded canonical quadratic gas,
with all its configured draws and component map.

1. With a fixed finite smooth cap $V>0$, take a consensus entering state
   $x_i=x_0$, $v_i=v_0\ne0$, with $x_0$ in the interior of the configured
   box when a box is present. All actual positive fitnesses are equal,
   so living clone acceptances are exactly zero. For the actual
   $h\downarrow0$ family with all other numeric parameters fixed,
   $v_i^+\to C_V(v_0)$ in probability. Thus the full native one-step
   kernel is not infinitesimal at that actual state.

2. With fixed saturation $s_c>0$, $N=2$, positive reward exponent and
   the quadratic reference reward orientation maximizing $-|x|^2/2$,
   take $x_i=.5e_1$, $x_j=.5e_1+.25e_2$, $v_i=v_j=0$.
   The reciprocal companion choices are the actual deterministic
   no-self rule, their diversity values are equal, and $f_i>f_j$.
   The lower-fitness row $j$ has fixed strictly positive acceptance
   $p=\min(1,(f_i-f_j)/[(f_j+\epsilon_c)s_c])$.
   Its position has an order-one accepted-copy/jitter displacement
   with probability bounded below independently of $h$.
   This remains true with the fixed reference cap because its
   position is committed before the cap.

Consequently the corresponding fixed-initial-state coordinate paths
$S_h(t)=S_{\lfloor t/h\rfloor}$ are not tight at zero in the usual
cadlag path topology on the complete coordinates. These assertions
concern an ACTUAL family of existing updates and its actual initial
states. They do not exclude a different scaling, a time-grid theory
at fixed $h$, a stationary boundary layer, or a reduced continuum
observable whose limit can be separately proved.
:::

:::{prf:proof}
At consensus the global standardization numerator is zero for both
measurement channels, and all maps and fitnesses agree. The exact
gate probability is zero. The common B1 velocity differs from $v_0$
by $O(h)$, the O noise tends to zero in every moment, both A
increments tend to zero, and B2 also changes velocity by a quantity
tending to zero. The force growth bound in the preceding proof gives
this conclusion with every actual Gaussian draw. A terminal box is
passed with probability tending to one from the interior.
Continuity of the actual cap therefore gives the first convergence.
The displacement has positive length
$|v_0|-|C_V(v_0)|=|v_0|^2/(V+|v_0|)$.

For the second statement, the quadratic rewards differ strictly, and
the strictly increasing reference logistic map and positive exponent
preserve their ranking after their common positive standardization.
Their pair distances and diversity maps agree. Thus $i$ persists
and $j$ accepts $i$ with probability $p>0$.
After acceptance $x_j^c=x_i+\sigma_JJ$.
For any sufficiently small fixed $\delta>0$, the event
$|x_i+\sigma_JJ-x_j|>2\delta$ has positive probability; for a box,
intersect it with a fixed interior ball about $x_i$ whose distance
from $x_j$ exceeds $2\delta$. This intersection still has positive
Gaussian probability when $\sigma_J>0$. For $\sigma_J=0$ choose
$\delta<|x_i-x_j|/2$. The configured collision acts on two zero
frozen velocities, so both remain zero. On the displayed bounded
jitter event the original kinetic position increment tends to zero
in probability. It follows that
$\liminf_{h\downarrow0}\Pr(|x_j^+-x_j|>\delta)>0$.
No jitter or kinetic Gaussian is removed from the update.

In either case a positive first-step displacement occurs at a time
$h\downarrow0$ from the fixed entering state. Cadlag paths are
right-continuous at zero, and the initial-value map is continuous
for the usual Skorokhod topology. A tight sequence with this fixed
initial value would have vanishing probability of such a
first-step displacement near zero: on any compact family of cadlag
paths, right continuity at zero is uniform. The proved lower bound
contradicts that necessary condition.
:::

:::{prf:remark} Absorbing marks and mandatory revival in a continuum identification
:label: rem-ntg-terminal-mark-regimes

The positive generator theorem uses the existing unbounded boundary
tag and therefore has no revival events. It does not replace the
terminal-box reference by that tag. In the reference's full marked
state, a retained dead row has mandatory revival in the very next
update, independently of the weak living acceptance scaling.
A state with that row outside $D$ at positive distance and a current
living donor therefore has an order-one next-step state change with
positive probability even when $s_c=s_0/h$. The same interior
Gaussian-jitter event in the proof gives this assertion.

A continuum marked process for the terminal reference must accordingly
derive its boundary/revival rule, including the retained dead
coordinates and the possible collapse of their holding time. The
smooth interior generator is insufficient. The fixed-step QSD,
stationary-law, native centroid-process and face-action results
already proved keep their actual grid time and terminal conventions.
They do not claim an infinitesimal full-state kernel in the
fixed-saturation/fixed-cap family tested above.
:::
