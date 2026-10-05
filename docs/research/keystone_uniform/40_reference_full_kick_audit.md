# Independent audit of the complete reference count-kick block

## 1. Reviewed revision and conclusion

The complete source `38_reference_full_kick_block.md` passes independent
review at SHA-256
`599f7f1bc4a56d7fc8dbb57343090e612a3a19ff1aa169557b645c1ec44a52b3`.
The first reviewed revision was
`e00bbf675813047da89faf85012bfd6047fbef0fbafa5efcaf068dfa74c01401`.
The final source adds the explicit conditional finite-array formula
(RFK.11f), retaining the subsequent mixed preparation expectation.
Its principal contraction and population radial estimates are unchanged.

The source proves the following distinct interfaces at the harmonic
reference step and viscosity. Their composition into a complete nonlinear
marked-law gap remains an additional estimate.

| Interface | Audited conclusion | Scope |
|---|---|---|
| Complete principal harmonic/cap map | Strict anisotropic contraction | Arbitrary self-adjoint alignment operators with spectrum in $[.994,1]$, without commutation |
| Actual own-graph differential | Exact spatial-force decomposition and signed upper account | Both actual count graphs, their own kernels and joint OU stage |
| Spatial pair consumers | Correct weighted forms against alignment | Root displacement and velocity correlations retained |
| Fresh OU defect | Correct complete Gaussian budget | Population pair law; finite arrays conditional on their preparation |
| Source-plan product | Exact Gaussian cross and fourth moments | Complete paired plans precede shared fresh recipient jitter |
| Radial B2 consumer | Explicit full-tail proportional bound | Population carrier with declared post-burn velocity moment |
| Full default marked law | Still requires closure | Signed spatial absorption, preparation change, revival and terminal conditioning |

## 2. Cap majorant and exact matrix certificate

For any self-adjoint $0\le D\le I$, put $C=DZ$ and $E=Z-C$.
Spectral calculus gives $\|Z\|^2-\|C\|^2\ge\|E\|^2$.
Completing the square in the cap residual gives precisely

$$
Q_\beta(R,C)\le\|R\|^2+\|Z+\beta R\|^2.
$$

This inequality is a complete-map majorant. It requires no standalone
claim that the cap contracts the signed quadratic. For the actual native
cap, its Jacobian is a self-adjoint multiplication operator with radial
and tangential eigenvalues in $[0,1]$.

The matrix in (RFK.2) is the exact deficit between the entering
$Q_\beta$ and this harmonic majorant. Its second rank-one term uses
the coefficients of $Z_0+\beta R$, namely
$f=\beta a_x-r_H$ and $g=v_H+\beta b$. Independent exact Python
`Fraction` arithmetic with the stated rational intervals gives the
following conservative bounds for the matrix after subtracting
$\operatorname{diag}(.0015,.073)$:

$$
M_{11}>.0000667701760982,\qquad
M_{22}>.0008170854600704,\qquad
|M_{12}|<.0000925683250136.
$$

These verify the source's weaker bounds $.00006$, $.0008$ and $.0001$,
including a strictly positive determinant. The interval certificate also
verifies $a_x^2+b^2<1$. Applying this two-dimensional quadratic to
each coordinate proves its stated real-Hilbert-space extension.

## 3. Both principal alignment operators without commutation

For a self-adjoint $A$ with $(1-a)I\le A\le I$, its defect
$e=(I-A)v$ satisfies $\langle e,v\rangle\ge\|e\|^2/a$.
The source applies this inequality separately to the two alignment
operators. It never changes their order or diagonalizes them in a
common basis.

At the second operator, the majorant defect is bounded by
$-k\|e_2\|^2-2(\beta-t)\langle R,e_2\rangle$, where
$k=(2-a)/a$. Completing its square leaves
$\delta(\beta-t)^2\|R\|^2$ with $\delta=a/(2-a)$.
The first-operator input quadratic similarly leaves
$\delta\beta^2\|r\|^2$.

The elementary bounds $\|R\|^2\le\|r\|^2+\|p_1\|^2$ and
$\|p_1\|\ge(1-a)\|p\|$ then give the exact two loss expressions
in the source. Their independently checked rational values satisfy

$$
\kappa_x-\delta[\beta^2+(\beta-t)^2]
>.0014939819458375>.00149,
$$
$$
[\kappa_v-\delta(\beta-t)^2](1-a)^2
>.0721254387891675>.0721.
$$

The upper quadratic norm comparison gives the stated scalar contraction
factor. For actual count operators, pair symmetry proves
$0\le L_j\le I$ on either the normalized array space or the lifted
population probability space. The noisy second operator may depend on
the OU array: the certificate holds for every operator satisfying its
spectral hypotheses. This verifies the claimed principal carrier, while
its own spatial derivative remains a separate force.

## 4. Exact own-graph differential and signed pair forms

Differentiating each actual count kernel before its kick gives
$\dot U=A_1p+aB_1$ and $\dot z=A_2\dot w-t\dot y+aB_2$.
The shared OU and final innovations have zero comparison derivative.
The affine stages therefore produce (RFK.5) exactly. The intermediate
providers are comparison laws; both endpoints retain their actual own
providers.

Applying the cap majorant and expanding its second-kick square preserves
$-2aD_2$, $-2a(\beta-t)E_2$, its Laplacian square and the full
$B_2$ cross term. Applying the harmonic matrix inequality to
$(r,p_1+aB_1)$ then preserves the first-kick alignment and signed
$E_1$ term as well. This checks every term and sign in (RFK.6).

Conditional weighted Cauchy--Schwarz gives $\|B_j\|^2\le2S_j$.
Pair symmetrization gives the arbitrary test-field bound in (RFK.8).
For the first consumer, the additional $a\sqrt2$ coefficient follows
from $\|L_1p\|\le\sqrt{D_1}$ and the preceding $B_1$ norm bound.
For the second consumer, $A_2$ commutes with its own $L_2$ because
$A_2=I-aL_2$; its contraction in that seminorm is valid. This step
does not assert commutation between $A_1$ and $A_2$.

The Gaussian source-box tails and bounded prepared velocities supply the
moments needed for differentiation. The spatial forcing retains its
actual displacement/velocity products. No averaged speed is substituted
for a correlated local factor.

## 5. Fresh OU noise and finite conditional preparation

The bound $|\nabla K|^2/K\le2/e$ is applied before conditioning on
the OU array. After it removes the kernel derivative from the integrand,
the prepared $R$ is independent of the fresh OU innovation. Each distinct
pair has exact noise-difference covariance $2q^2I_d$. Diagonal finite
pairs have zero displacement and contribute zero. This proves (RFK.10)
without a false graph/noise independence assumption.

The population pair expansion in (RFK.11) factors only independent
copies of its deterministic root law. The final finite formula
(RFK.11f) is conditioned on the realized complete preparation; its
two moment factors are empirical averages of that same array. If the
preparation is subsequently random, their product remains inside its
expectation. This clarification prevents an invalid replacement by two
unconditional averaged moments. The local mixed product is retained in
both carriers.

## 6. Full-tail source product and radial population consumer

Expanding $|S_\theta+I_\theta J|^2|\delta S+\delta I J|^2$
checks the exact source product (RFK.12). In particular, its Gaussian
linear cross product is
$4I_\theta\delta I\sigma_J^2S_\theta\cdot\delta S$ and its
fourth-moment term is $I_\theta^2\delta I^2d(d+2)\sigma_J^4$.
Bounding the displayed cross term yields the stated coefficients
$12.05$ and $.6015$. The exact fresh-jitter displacement variance gives
$d_X^2=\mathbb E|\delta S|^2+.03\mathbb E\delta I^2$, verifying
the factor $20.05$. Prepared velocity differences are fixed before this
jitter, so the separate $X_2d_P$ product bound is valid.

The root's first spatial forcing obeys its stated pointwise bound using
the actual environment second velocity moment. The principal first
operator's pointwise action retains its nonlocal incoming contribution
$ad_P$. These two estimates reproduce both $d_Y$ and the weighted
$H_Y$ product. The unweighted estimate uses the true $L^2$ count
contraction; the weighted one uses the exact conditional source product.

For the actual second provider, its vector moment is deterministic and
bounded by $M_w$. Rewriting the precap velocity and using
$|DC_V(z)z|\le V/4$ gives the pointwise radial bound on
$|DC_V(z)w|$. Its full-tail source-weighted product is then bounded by
$S_0d_Y+S_xH_Y$. Its plain $L^2$ bound uses the actual alignment
energy and the declared all-slot post-burn moment.

In the remaining environment term, Cauchy--Schwarz bounds the joint
product of $R'$ and the uncapped $w'$ by $\|R\|_2M_w$.
The root's correlated radial factor stays inside its norm. These steps
check the first inequality in (RFK.14) without factoring source, OU
velocity or landing position. Exact rational substitution gives upper
coefficients less than

$$
.009398092484727<.0094,\qquad
.000365530607129<.000366.
$$

The intermediate $M_w=.70$ envelope and both bounds in (RFK.15) also
check. This radial theorem is a population statement. A finite random
empirical velocity budget would need its own mixed-product argument,
as the source explicitly states. No jitter or OU event is removed and
no fixed additive tail floor is dropped.

## 7. Endpoint and structural checks

The source proves the complete principal contraction and the actual
spatial-force interfaces. Its full signed absorption remains open: the
absolute radial coefficient alone does not fit into the principal
norm-contraction margin. The changing rooted preparation, mandatory-dead
feedback, terminal marking and actual alive/survival normalizations are
also retained as separate obligations. The harmonic formulas provide no
Rastrigin certificate. No default invariant, QSD or full active uniform
law rate is claimed.

Verification included complete proof reading, an independent principal
derivation without operator commutation, exact rational matrix and radial
coefficient checks, and structural checks. The accepted source contains
nine unique formal labels and sixteen unique equation tags, with balanced
formal directives and display-math delimiters and no trailing whitespace.

## 8. Zero-displacement precision revision

The accepted source was subsequently revised to SHA-256
`ae857ee67f7c1d44bbc2b3dd88cf9bf61ac3e1fdcae0d00dbe6b244dd25ac207`.
The final norm bound in (RFK.14) now reads
$\le .0094d_X+.000366d_P$ instead of a strict inequality.
This includes the zero-displacement case. Both strict scalar coefficient
certificates, the conditional finite formula (RFK.11f), proof and scope
are unchanged. The historical accepted revisions above remain recorded;
this precise non-strict endpoint repair passes review as well.
