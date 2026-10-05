# Independent audit of the original-jitter and conditional cap consumers

(sec-jca75-source)=
## 1. Frozen exact-jitter source and accepted scope

:::{prf:definition} Original-jitter review record
:label: def-jca75-source

The independently reviewed source is
`73_default_first_jitter_tensor.md` at SHA-256
`3d8338a7cd399f9d746ae6b9bac8e8f847fdb2ba33cec83be235ba46e022a67a`.
The accepted inputs are the already reviewed research68 signed
ledger and research69 conditional cap account, at their recorded
hashes. This review does not reopen their completed proofs.

The new result integrates the entire original recipient-jitter
contribution to the first signed count response, including its
force square. It also proves a matched-status inward-source sign
for the linear pair term. Neither result is a general negative
first-response bound, an alive-law estimate or an invariant class.
:::

(sec-jca75-conditioning)=
## 2. Actual preparation conditioning and covariance degeneracies

:::{prf:lemma} Coupled source-plan and Gaussian-tilt audit
:label: lem-jca75-conditioning

The source register and Gaussian tilt (FJT.1)--(FJT.6) are
valid for every stated pair of copy statuses and interpolation
parameter, including zero-variance and singular cases.
:::

:::{prf:proof}
The coupling freezes both complete accepted preparation plans
before drawing recipient jitters. This fixes each actual source
mean, copied/revived indicator and original-slot component/Haar
velocity. The common recipient jitter preserves each endpoint
transition marginal, and distinct recipient indices retain
independent jitters. Sharing a donor does not identify their
recipient Gaussian variables. The interpolation is explicitly a
comparison path rather than an asserted intermediate preparation.

For a distinct pair, affine expansion gives exactly
$$
v=\sigma_J^2(I_i^2+I_j^2),\quad
w=\sigma_J^2(j_i^2+j_j^2),\quad
k=\sigma_J^2(I_ij_i+I_jj_j).
$$
Their joint covariance is positive semidefinite, hence
$k^2\le vw$. The declared bounds on the indicators give all
scalar endpoints in (FJT.5).

For a general semidefinite joint Gaussian, add an independent
standard Gaussian observation noise to its position vector.
The observation covariance $G=I+V$ is strictly positive.
Conditioning the observation to zero gives the tilt mass
and linear-regression means in (FJT.3)--(FJT.4). Regression
residuals are jointly Gaussian and independent of the
observation because their cross covariance vanishes. This
argument does not invert the original singular covariance.
The resulting cross block is
$(I-VG^{-1})T=G^{-1}T$. Scalar specialization yields precisely
$\alpha=v/(1+v)$, $\chi=k/(1+v)$ and
$\omega=w-k^2/(1+v)$.
In particular $v=0$ implies $k=0$ without requiring $w=0$.
No division by jitter variance or positive copied mass is used.

All these conditionings precede the new jitter and future
terminal marks. The source explicitly does not apply these
Gaussian formulas after conditioning on survival.
:::

(sec-jca75-pair)=
## 3. Complete linear response and spatial-defect moments

:::{prf:lemma} Pair-moment and signed-completion audit
:label: lem-jca75-pair

Equations (FJT.7)--(FJT.11) are exact conditional moments of
the source's joint Gaussian pair and correctly retain all
source/component velocity correlations.
:::

:::{prf:proof}
Write the tilted pair as $D=\mu+\xi$, $\eta=n+\zeta$.
Then
$$
\mathbb E(D\cdot\eta)=\mu\cdot n+d\chi=s,\qquad
\mathbb E[D\eta^{\mathsf T}]=\mu n^{\mathsf T}+\chi I_d.
$$
The third moment has exactly the three contributions
$ns$, $\chi n$ and $\omega\mu$, so it equals
$h=n(s+\chi)+\omega\mu$. Multiplication by the tilt
mass gives (FJT.8). Distributing
$\Psi_1=A_r\eta+A_p\pi$ reproduces every term of
(FJT.9), including the negative $A_rn\cdot\pi$.

The deterministic square completion in $\pi$ gives
$$
-A_p|\pi|^2+\pi\cdot(A_psH-A_rn)
=-A_p|\pi-(sH-A_rn/A_p)/2|^2
 +\frac{A_p}{4}|sH-A_rn/A_p|^2.
$$
Combining its mixed term with $A_rH\cdot h$ gives
$A_rH\cdot[(s/2+\chi)n+\omega\mu]$, with the sign and
coefficient stated in (FJT.10).

The variance of $D\cdot\eta$ is
$$
\omega|\mu|^2+\alpha|n|^2+2\chi\mu\cdot n
 +d(\alpha\omega+\chi^2).
$$
The centered linear and centered bilinear parts have zero
covariance by odd Gaussian moments. The two remaining
fourth-moment pairings give the last term above.
Adding $s^2$ proves (FJT.11). The relation to the prior
$S_\theta$ has its correct factor $1/2$ because
$\dot k^2/k=K(D)(D\cdot\eta)^2$.
Every velocity factor in these conditional identities is
fixed by the full pair plans, rather than independent of
their source displacement. No unconditional moment product
is introduced.
:::

(sec-jca75-triple)=
## 4. Exact force square and coincident finite environments

:::{prf:lemma} Two-edge Wick contractions and normalization audit
:label: lem-jca75-triple

The two-edge formulas (FJT.12)--(FJT.16) give the entire
first-provider square with its actual shared query jitter,
finite diagonal environment terms and original count denominators.
:::

:::{prf:proof}
The coefficient-matrix construction in (FJT.14) associates
one independent Gaussian with each distinct root index.
When the two environment indices coincide, the rows share
that same Gaussian; the resulting singular covariance is
permitted by the preceding observation tilt. A query self
edge has zero force and is removed only on that basis.

For the tilted two-edge pair, the covariance of the centered
linear scalar parts gives the four contractions
$$
\mu_1^{\mathsf T}B_{12}\mu_2,\quad
n_1^{\mathsf T}A_{12}n_2,\quad
\mu_1^{\mathsf T}C_{21}^{\mathsf T}n_2,\quad
n_1^{\mathsf T}C_{12}\mu_2.
$$
The two cross-edge fourth-moment pairings are
$\operatorname{tr}(A_{12}B_{12}^{\mathsf T})$
and $\operatorname{tr}(C_{12}C_{21})$.
These orientations agree with the covariance definition
$C_{\ell m}=\operatorname{Cov}(D_\ell,\eta_m)$.
The within-edge pairing is already in $s_1s_2$.
Thus (FJT.12) is the complete scalar product moment.
Distributing the two force vectors gives the four terms
of $\mathcal T_{12}$ in (FJT.13).

The linear part of the accepted first ledger is the exact
$N^{-2}$ pair average, multiplied by $a$. Squaring the
$N^{-1}$ environment force and then taking the root
$N^{-1}$ average gives its exact $N^{-3}$ triple sum,
multiplied by $a^2A_p$. This includes $j=k$, even though
$i=j$ or $i=k$ yields a zero force. The population square
uses two independent environment integrals conditional on
the same query root and therefore has the same two-edge
construction. These are precisely the two terms of (FJT.15).

The source means and velocities are bounded, and the remaining
displacements are affine Gaussians with bounded coefficients.
Alternatively $K(D)|D|\le e^{-1/2}$ directly gives $L^2$
forces. Their triple products are absolutely integrable by
Cauchy--Schwarz. Fubini therefore applies to the square and
its conditional integration without removing any tail.
:::

(sec-jca75-sign)=
## 5. Inward-source sign and the full-law boundary

:::{prf:lemma} Accepted signed corollary and precise remaining account
:label: lem-jca75-sign

The linear sign corollary (FJT.17)--(FJT.18) is nonempty
at the actual preparation interface and has the scope stated
in the source. Equations (FJT.19)--(FJT.20) correctly preserve
the unresolved complete absorption and alive-law obligations.
:::

:::{prf:proof}
Matched endpoint statuses give $j_i=j_j=0$, hence
$n=e$, $\chi=\omega=0$ and $s=\Delta\cdot e/\Gamma$.
Substituting $H=-\lambda\Delta$ and $\pi=-\lambda e$
into the full linear response gives exactly
$$
\mathcal L=\lambda(A_r-A_p\lambda)
 \left[|e|^2-\frac{(\Delta\cdot e)^2}{\Gamma}\right].
$$
The bracket is nonnegative whenever $|\Delta|^2\le\Gamma$.
The accepted coefficient bounds give
$A_r-A_p/20<-.0062925$, so the signed scalar is negative
throughout the stated $\lambda$ interval. Zero displacement
is included by the non-strict final inequality.

For the concrete two-slot symmetric inward family, both
alive rewards and both nonself diversity distances are tied.
The active gate is therefore zero, the components are
singletons and the prepared velocities retain the original
inward relation. Its source pair separation is at most $.6$,
so the required size condition holds along interpolation.
This proves the stated actual nonconstant-velocity example.
It does not assert that a general accepted Haar or revival
plan obeys that relation.

The positive force square remains in (FJT.15), and the full
second response uses the actual jitter-dependent first output.
The source assigns no favorable sign to that response or
to mismatched-status covariance terms. Its final proposed
absorption inequality is explicitly unproved. Preparation,
terminal marks, each own survival denominator and invariant
class or delayed-block control remain separate requirements.
No full default convergence rate, survivor mixing estimate or
finite-swarm quasi-stationary rate follows from this audit.
:::

(sec-jca75-result)=
## 6. Review result

:::{prf:remark} Accepted original-jitter revision
:label: rem-jca75-result

The frozen source73 in Section 1 passes this independent
mathematical review without requiring a source correction.
Its formulas consume all original recipient Gaussian outcomes
and preserve the actual root, pair and triple finite sums.
The accepted endpoint is the exact first-provider signed
account and the restricted linear-sign criterion. The review
does not certify the remaining inequality (FJT.20).
:::

(sec-jca75-cap-source)=
## 7. Frozen conditional-cap source and its actual hypotheses

:::{prf:definition} Conditional cap review record
:label: def-jca75-cap-source

The independently reviewed additional source is
`74_default_general_cap_absorption.md` at SHA-256
`fe58c2ccd64571ac2979bd3d36877f2aef0c6528e5d234cd13edd77cca0b5225`.
It proves a conditional actual native-cap squared-derivative
coefficient $159/200$ for vectors fixed before the new OU.

The population premise is its actual deterministic joint OU
provider first velocity moment at most $.70$. The finite
premises are the complete prepared array budgets
$|P_i|\le4$, $\langle|P|^2\rangle_N\le.56^2$ and
$\langle|X|^2\rangle_N\le12.25$. They are imposed before
the new original OU array. The $N=1$ assertion instead
uses its identically zero count field. The result neither
conditions a row Gaussian on an empirical-OU event nor
uses these moments as pointwise speed or position bounds.
:::

(sec-jca75-cap-reduction)=
## 8. Actual own graph and finite exceptional-provider charge

:::{prf:lemma} Bare Gaussian reduction and empirical-provider audit
:label: lem-jca75-cap-reduction

The reduction (GCA74.5)--(GCA74.7) retains the actual
second count graph and its correlated OU velocity numerator.
Its finite exceptional-provider probability is valid under
the entire pre-OU conditional Gaussian array.
:::

:::{prf:proof}
The exact coefficients satisfy
$r_H=mb=t(c+a_x)$ and $z_v=c-tb$. Thus
$$
z_0=mw-tx_1=\mu+mq\xi,\qquad
\mu=z_vU-r_HX,\qquad
x_1=(cU-\mu)/b.
$$
The last identity follows directly from
$c-z_v=tb$ and $r_H=mb$. It retains the unbounded
prepared position through $\mu$.

For the actual joint count provider,
$z=(m-ad_y)w-tx_1+aM_y$ with
$0\le d_y\le1$. Its own numerator satisfies
$|M_y|\le K$ whenever the population first moment or
actual empirical first moment is at most $K$.
The included finite self terms cancel exactly in $L_yw$.
The event $|z|\le r$ therefore implies
$$
(1-a/m)|z_0|
\le r+\frac{at}{mb}(c|U|+|\mu|)+aK.
$$
Since the actual first count update is convex,
$|U|\le4$. This is precisely (GCA74.5); no second
graph, velocity or Gaussian is replaced by its expectation.

For a fixed eligible finite preparation, self-adjoint count
contraction gives $\|U\|_{2,N}\le.56$, so the OU mean
array has RMS at most $c(.56+.02\sqrt{12.25})<.606$.
The original independent Gaussian array and $q<.198$
give the deterministic outcome bound
$$
\langle|w|\rangle_N
\le.606+.198\sqrt{\langle|\xi|^2\rangle_N}.
$$
Thus a provider first moment above $1.5$ implies
$\sum_i|\xi_i|^2>20N$, since $(149/33)^2>20$.
The exact chi-square moment generating function at
$1/4$ is $2^{3N/2}$. Markov's inequality yields
$e^{-5N}2^{3N/2}\le2^{-5N}$ because
$(13/2)\log2<5$.

This charge is unconditional on that new Gaussian
provider event, conditional only on the complete
preparation. It can be added to a root's unconditional
Gaussian small-ball bound even when that root and
its own empirical graph are correlated.
The population post-burn premise also checks:
first count contraction and source second moment give
$$
\mathbb E|w|^2
\le c^2(.55+.02\sqrt{12.03})^2+3q^2<.70^2.
$$
Only its actual provider first moment is inferred by
Cauchy--Schwarz.
:::

(sec-jca75-cap-thresholds)=
## 9. Shifted Gaussian balls and the exact scalar deficit

:::{prf:lemma} Threshold register and rational cap-defect audit
:label: lem-jca75-cap-thresholds

Every probability ceiling in (GCA74.8)--(GCA74.11) is
valid under its stated conditional register, and the
resulting cap defect exceeds $41/200$.
:::

:::{prf:proof}
The retained exponential interval yields
$\sigma>.195$, $\sigma^2<.0392$,
$\alpha>.993$, $\delta<.003063$,
$\delta cV_c<.012$ and $B<.0031$.
Direct rational arithmetic also gives
$.021+.0031(.993)<.025$.
These verify the common radius bound
$A_r+B<(r+.025)/.993$, which is valid with the
larger finite provider threshold $1.5$.

For $|\mu|\le1$, the reduced event belongs to an
isotropic Gaussian ball with that common radius.
The Gaussian mass of a ball is maximal at zero shift:
after rotating the shift to one coordinate, integrate
symmetric interval slices whose one-dimensional
translated Gaussian masses decrease for a positive
shift. This proof applies to every radius and needs
no bounded individual position.

The centered three-dimensional radial mass is at most
$\frac45\int_0^{R_j}u^2e^{-u^2/2}\,du$.
Taylor's formula has negative remainder after the even
degree-$30$ expansion of $e^{-v}$ for every $v\ge0$,
giving exactly the rational polynomial $G_j$.
For $|\mu|>1$, the scalar projection of the reduced
event implies a one-dimensional Gaussian below
$-[1-(r_j+.025)/.993]/\sigma$.
The numerator remains positive through $r_{12}=.60$.
Markov's inequality for its square gives the stated
rational $T_j$.

The maximum of these bounds covers both mean regimes.
For finite $N\ge2$, adding the complete exceptional
provider charge gives at most
$\max(G_j,T_j)+1/1024$. It does not use a row
Gaussian conditional on the provider event.
For $N=1$, $z=z_0$ and the smaller centered-ball
radius gives a bound at most $G_j$.

Independent exact `Fraction` evaluation verified
all twelve comparisons against the stated $u_j$.
It also verified
$$
.2057<
\sum_{j=1}^{12}[H(r_j)-H(r_{j-1})](1-u_j)
<.2058,\qquad
H(r)=1-\left(\frac2{2+r}\right)^2.
$$
The finite lower step function in (GCA74.14) is
pointwise below $H(|z|)$, including the whole tail
above $.60$. Thus the conditional expected defect
exceeds $.205=41/200$.
The native cap operator norm is $2/(2+|z|)$.
Multiplication by a pre-OU-fixed test-vector square
proves (GCA74.12)--(GCA74.13), with non-strict
vector inequalities also covering the zero vector.
:::

(sec-jca75-cap-account)=
## 10. Correlated second-force terms and each own restriction

:::{prf:lemma} Full cap expansion and weighted own-normalization audit
:label: lem-jca75-cap-account

Equations (GCA74.15)--(GCA74.17) retain every correlated
second-force term and the required weighted exceptional
and own-survival losses.
:::

:::{prf:proof}
Before fresh OU, the complete first derivative fixes
$R=a_xr+bE$ and $Z_b=z_vE-r_Hr$.
The full second derivative is $Z=Z_b+aF_2$.
Expanding
$$
\mathcal L_C
=\langle Z,(I-D^2)Z\rangle
 +2\beta\langle R,(I-D)Z\rangle+\beta^2|R|^2
$$
gives exactly (GCA74.15). Only the first bare square
uses the conditional fixed-vector defect bound.
Both the mixed term involving $F_2$ and its complete
quadratic remain under their actual joint OU and
own-provider expectation.

For the bare capped phase quadratic, direct expansion
gives (GCA74.16). Conditional variance proves
$A^2\le B$ because $D$ and $A=\mathbb ED$ are
self-adjoint and
$|Av|^2\le\mathbb E|Dv|^2$ for every fixed vector.
No commutation between different noisy Jacobians
or scalar replacement of $A$ is used.

Let $\kappa=41/200$ and
$\mathcal D(T)=|T|^2-|DT|^2$.
The raw proposal gives
$$
\mathbb E\mathcal D(T)
\ge\kappa\mathbb E[|T|^2\mathbf1_{\mathcal G}],
\qquad 0\le\mathcal D(T)\le|T|^2.
$$
Subtracting its actual restriction complement yields
$$
p_{\mathcal S}\mathbb E[\mathcal D(T)\mid\mathcal S]
\ge\kappa p_{\mathcal S}
       \mathbb E[|T|^2\mid\mathcal S]
 -(1-\kappa)\mathbb E[|T|^2\mathbf1_{\mathcal S^c}]
 -\kappa\mathbb E[|T|^2\mathbf1_{\mathcal G^c}].
$$
Division by that proposal's own positive probability
is exactly (GCA74.17). Both lost moments remain
weighted by the actual correlated test vector.
The source correctly declines to replace them by an
unconditional probability floor times a test-vector
moment. In particular global extinction estimates
before recipient jitter do not imply uniform
extinction bounds conditional on every prepared array.
No conditioned innovation is treated as fresh Gaussian.
:::

(sec-jca75-cap-result)=
## 11. Accepted conditional coefficient and unchanged missing gap

:::{prf:remark} Conditional cap revision passes
:label: rem-jca75-cap-result

The frozen source74 in Section 7 passes this independent
review without a source correction. Its exact rational
certificate was rerun, and the Gaussian reduction,
finite chi-square charge, matrix expansion and own
restriction algebra were independently checked.

The accepted endpoint is a uniform conditional
squared-derivative deficit for pre-OU-fixed vectors
under the declared actual provider or prepared-array
budgets. This is not an unweighted contraction of the
OU-dependent second force. Its signed mixed force,
first response, preparation/component/Haar/revival
changes and weighted readout losses remain in the
full account. No general default law gap, invariant
comparison class, finite survivor convergence rate or
$N$-independent exact QSD rate is certified
by this accepted intermediate result.
:::
