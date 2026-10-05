# Independent audit of the reference harmonic velocity burn

## 1. Reviewed source and decision

The complete proof in `34_reference_velocity_burn.md` passes independent
review at SHA-256
`800da5a42259e113af22c148196d72e62d86bce1d4a19e4bf3c06f9fe2ab0020`.
The earlier revision
`c62f7bff3f75cd596703d5769be5fa040011a85f664b9baf39d7582a34e62e50`
was reviewed first. The final revision clarifies the differential and
finite-segment meanings of (RVB.13); the moment estimates and complete
fixed-root provider theorem are unchanged.

This review checks the actual harmonic count branch at $h=.04$, $\nu=.3$
and $L=2$, including mandatory revival, original-slot component collision,
recipient jitter, both count kicks, uncapped Gaussian innovations, the
native cap and terminal marking. The finite-chain moment statements and
the population-provider statements have distinct carriers.

| Result | Audited conclusion | Scope |
|---|---|---|
| Preparation and precap energy | Correct averaged inequalities | All source/component outcomes; no entering dead-position moment |
| Population velocity burn | Full-slot RMS speed at most $.55$ after six updates | Actual marked population recursion under its declared rooted consistency hypotheses |
| Finite survivor velocity burn | Full-slot RMS speed at most $.56$ after six updates | Each killed chain's own current-survivor law, provided $e_N\le.01$ |
| Joint OU moment | Population bound $<.69$, proposed finite-stage bound $<.70$ | Averaged joint stage laws, without a realized maximum or future-survival conditioning |
| Full-jitter provider feedback | Complete fixed-root physical, marking and alive-normalized bounds | Both deterministic population providers may change; root preparation law is held fixed |
| Full default law convergence | Remains open | Preparation-law change, correlated displacement and marked feedback still require a signed block estimate |

## 2. Preparation and both actual count kicks

The component calculation uses the original frozen velocities. On each
actual component, subtracting its mean leaves zero total centered velocity,
so the cross term vanishes for every common orthogonal Haar mark. The
prepared energy is exactly the component mean energy plus
$\alpha_{\rm col}^2$ times its centered energy. Since
$|\alpha_{\rm col}|=.5$, the total component energy contracts before
expectation. Donor velocities are never substituted into this identity.

Every selected source lies in the alive box. Conditional on the discrete
plan, recipient jitter is independent and centered; therefore its source
cross term is exactly zero and its added second moment is
$I_i d\sigma_J^2$. This proves the uniform source budget $X_2^2=12.03$,
including revived slots with arbitrarily distant retained dead positions.
The first viscous matrix depends on these jitters. The later proof does
not discard the resulting $U$--$X$ correlation: it charges that cross term
by Cauchy--Schwarz.

For every realized first count graph, its symmetric Laplacian satisfies
$0\le L_X\le I$. Hence $I-t\nu L_X$ contracts energy when $t\nu\le1$.
The exact affine identity
$z_0=z_vU+z_rX+mq\xi$ has an independent subsequent OU innovation.
Only its OU cross terms are set to zero. Its prepared cross term retains
the full $(z_vr+|z_r|X_2)^2$ allowance.

At the second kick, the reviewed energy identity in
{prf:ref}`thm-rcb-finite-dissipation` holds pointwise for the actual noisy
graph. Young's inequality in its Laplacian quadratic form leaves a
nonpositive alignment term and the positive budget
$2t^3\nu\langle y,L_yy\rangle/(2-t\nu)$. The Gaussian pair kernel
gives $\langle y,L_yy\rangle\le1/e$ for every $y$.
Thus dropping the alignment term is valid and keeps the full positive
second-graph remainder. No independence of this graph and OU innovations
is assumed.

The population energy import requires the declared canonical uniform-root
finite-swarm limit and almost surely finite rooted components. Original
and prepared squared speeds are bounded, allowing the finite averaged
collision inequality to pass through that consistency result. On the
actual prepared and joint stage laws the count operators remain
self-adjoint positive contractions; their pair integrals reproduce the
finite energy argument. These hypotheses are stated in
{prf:ref}`def-rvb-record` and are not inferred from an independent-output
collision model.

## 3. Cap concavity and the exact six-update certificates

The native squared-radius map
$f_V(u)=u/(1+\sqrt u/V)^2$ has positive first derivative and negative
second derivative on $(0,\infty)$, with continuous concave extension to
zero. Jensen applies to the joint probability obtained by drawing a
uniform slot and then all actual innovations. It produces an averaged
speed recursion, rather than a reduced rowwise cap.

The primitive rational bounds $z_v<.961$, $F_{\rm box}<.137$ and
$B_{\rm vel}<.118$ are valid. Each burn certificate was independently
checked with exact Python `Fraction` arithmetic by comparing

$$
(.961a+.137)^2+.118
<\left(\frac{2b_*}{2-b_*}\right)^2.
$$

The six population transitions are
$2\to1.022\to.739\to.628\to.580\to.559\to.55$.
The preservation certificate $.55\to.545$ and the strict fixed-point
certificate $.545\to.545$ also pass. The scalar comparison map has
derivative bounded by $z_v<1$, so a population fixed point's necessary
moment inequality indeed forces its RMS speed strictly below $.545$.
This implication does not establish existence of that fixed point.

For the finite killed chain, proposing its next update from its own
current-survivor law gives the same raw energy bound. Its one-step
survival probability is at least $1-e_N$. Dividing the nonnegative
output energy by that actual survival probability yields exactly
$r_{n+1}\le T(r_n)/\sqrt{1-e_N}$. This is a last-step conditional
normalization; no independent conditioning of intermediate innovations or
whole-history replacement is used.

When $e_N\le.01$, the factor is less than $1.01$. Exact square
certificates with threshold $b_*/1.01$ check all six finite transitions
$2\to1.032\to.750\to.639\to.591\to.570\to.560$ and preservation
at $.560$. The explicit population-size threshold follows directly from
$e_N=(1-a_{\rm ret})^N$. The QSD statement is a necessary moment bound
obtained by repeating that conditional recursion at a QSD, if one exists.

The two joint OU moment inequalities were also checked by exact rational
squaring. Their bounds retain the OU covariance and use the full-slot
energy from the actual entering law. They do not bound a particular
empirical provider after conditioning on its future survival.

## 4. Complete full-jitter fixed-root feedback

Every common source-box root has second positional moment at most
$X_2^2$, even when prepared under a different entering law. Each
interpolated first provider gives a convex average of velocities bounded
by $V_c$. Consequently its actual first-drift position has uniform
$L^2$ norm at most $mX_2+tV_c$.

At every second-provider interpolation, write its actual precap velocity
as $Z=l(y)w+B(y)$ with $l\ge A=m-t\nu$ and
$|B|\le t|x_1|+t\nu M_w$. The cap's radial identity
$|DC_V(Z)Z|\le V/4$ then gives the pointwise bound on
$|DC_V(Z)w|$. Minkowski converts this to the stated full-jitter
$L^2$ coefficient $\overline S_w<.58$. The jitter tail remains part of
that expectation; it creates no additive tail residual.

Differentiating at $y=x_1+tw$ retains the true landing-position/OU
correlation. The displayed averaged Jacobian constants and their exact
rational bounds pass. The saved correction to (RVB.13) distinguishes
the differential estimate from a finite perturbation, for which the
Jacobian must be integrated along the actual segment. An averaged
Jacobian cannot be multiplied by an arbitrary correlated $L^2$
displacement. The source explicitly keeps that restriction.

The complete provider theorem uses a stronger admissible input:
first-provider field variations are uniformly bounded by the deterministic
$d_0$. Every intermediate first provider retains the same source-box
moment. Its phase displacements can therefore be pulled outside each
Jacobian expectation, giving the claimed RMS coefficient. Changing the
second provider then leaves position unchanged and uses the full-jitter
radial estimate. Minkowski composes the two comparisons. Exact rational
squaring verifies $C_0<.00578$.

The shared OU and final innovations preserve both exact root-kernel
marginals. Terminal mark changes are controlled using the independent
final Gaussian's actual density and the two box faces in each coordinate.
The safe-return lemma applies to each frozen-provider kernel because its
first average is still pathwise bounded by $V_c$. Restricting the coupling
to common-alive mass and completing its residual after each own alive
normalization proves the stated conservative factor $2/a_{\rm ret}$.
Those actual alive masses are neither identified nor replaced by a
constant. This root alive-normalization comparison is distinct from
finite whole-swarm current-survival conditioning.

## 5. Remaining scope and verification

The completed interface holds the prepared root law fixed. Changing the
entering law also changes the rooted preparation, its realized fitness
statistics and mandatory-revival donor choices. The arbitrary correlated
displacement and signed phase block remain separate estimates. The proved
positive alive floor does not supply a small-dead invariant class at the
default box. These remaining terms are explicitly retained in the source.

The conclusions concern the harmonic force arm. Rastrigin's additional
force terms cannot import its velocity thresholds. No QSD existence,
fixed-point existence, full default convergence or impossibility theorem
is inferred from this moment/interface closure.

Verification used complete proof reading, exact rational burn/provider
certificates, checks of the stated theorem imports, and structural checks
of directives, labels and equation tags. No algorithm or formal manuscript
was edited during this independent review.
