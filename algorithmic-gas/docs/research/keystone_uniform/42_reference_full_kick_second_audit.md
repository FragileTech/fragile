# Second audit of the default full-kick intermediate estimates

(sec-rfsa-record)=
## 1. Exact retained source and audit endpoint

:::{prf:definition} Frozen full-kick source and scope
:label: def-rfsa-source

The audited source is
`docs/research/keystone_uniform/38_reference_full_kick_block.md`,
SHA-256
`599f7f1bc4a56d7fc8dbb57343090e612a3a19ff1aa169557b645c1ec44a52b3`.
It retains the actual harmonic count kicks at $h=.04$, $\nu=.3$,
$V=2$, $L=2$, source-box recipient jitter and full Gaussian OU/final
innovations. This audit establishes the mathematical applicability of
its stated principal, signed and source-product intermediate estimates.
It does not identify their remaining signed forcing with a proved
full active or marked contraction margin.
:::

(sec-rfsa-finite-products)=
## 2. The finite conditional mixed product in RFK.11f

:::{prf:lemma} Conditional finite second-graph product audit
:label: lem-rfsa-finite-product

In {prf:ref}`lem-rfk-fresh-ou-defect`, freeze the entire finite
coupled preparation, including all actual recipient jitters and first
count fields. The arrays $R_i=\dot y_i$ and
$\bar w_i=c(U_i-tX_i)$ are then deterministic. For the actual noisy
second kernel, its spatial defect satisfies
$$
\mathbb E_\xi S_2\le\frac8e\left\{
\langle|R|^2|\bar w|^2\rangle_N+
\langle|R|^2\rangle_N\langle|\bar w|^2\rangle_N\right\}
+\frac{4dq^2}{e}\langle|R|^2\rangle_N.
$$
This is valid for the actual globally correlated preparation and the
actual second graph. Averaging that preparation afterwards leaves the
second term as
$\mathbb E[\langle|R|^2\rangle_N\langle|\bar w|^2\rangle_N]$.
No product of its two unconditional expectations is inferred.
:::

:::{prf:proof}
The Gaussian count-kernel identity
$|\nabla K(z)|^2/K(z)\le2/e$ is pointwise, so it can be applied
before averaging the graph/OU correlation. It gives
$$
S_2\le\frac1{eN^2}\sum_{i,j}
                       |R_i-R_j|^2|w_i-w_j|^2.
$$
Each own finite marginal has independent fresh $\xi_i$ after the
frozen preparation. For $i\ne j$,
$$
\mathbb E_\xi|w_i-w_j|^2
=|\bar w_i-\bar w_j|^2+2dq^2.
$$
For $i=j$, $R_i-R_j=0$, so the same sum formula can harmlessly
use $2dq^2$ on its diagonal. Its mixed deterministic contribution is
bounded by
$$
\frac4{N^2}\sum_{i,j}
(|R_i|^2+|R_j|^2)(|\bar w_i|^2+|\bar w_j|^2)
=8\left\{\langle|R|^2|\bar w|^2\rangle_N+
 \langle|R|^2\rangle_N\langle|\bar w|^2\rangle_N\right\}.
$$
The noise contribution uses
$N^{-2}\sum_{i,j}|R_i-R_j|^2
=2[\langle|R|^2\rangle_N-|\langle R\rangle_N|^2]
\le2\langle|R|^2\rangle_N$.
These prove the displayed bound. They neither require independent
prepared rows nor replace the actual noisy count field by a frozen
population field. The distinction after further preparation averaging
is simply the law of conditional expectation; its product has no
independence premise.
:::

(sec-rfsa-population)=
## 3. Population source products and the radial consumer

:::{prf:lemma} Applicability of the population-only radial consumer
:label: lem-rfsa-population-consumer

The hypotheses and constants of
{prf:ref}`thm-rfk-own-second-cap-consumer` apply to a source-plan
coupling of two actual prepared population laws after the declared
velocity burn. They do not apply by replacing finite empirical
velocity moments with their unconditional RMS. Its source polynomial,
weighted $H_Y$ bound and $a\|DC_V(z)B_2\|_2$ estimate retain the
required root/environment correlations.
:::

:::{prf:proof}
The prepared interpolation has
$X_\theta=S_\theta+I_\theta J$, $S_\theta\in[-2,2]^3$,
$I_\theta\in[0,1]$, with its coupled velocities and phase
displacements fixed before fresh jitter. Gaussian second and fourth
moments give exactly (RFK.12), including the signed term
$4I_\theta\delta I\sigma_J^2S_\theta\cdot\delta S$.
Its bound gives coefficients $12.05$ and $.6015$, and
$\mathbb E|r|^2=\mathbb E|\delta S|^2+.03\mathbb E\delta I^2$
gives the uniform coefficient $20.05$. Multiplication by a
plan-measurable $p$ instead gives $\||X_\theta||p|\|_2\le X_2d_P$.

The first-force pointwise bound
$|B_1|\le\ell(V_c+r_0)(|r|+d_X)$ uses Cauchy--Schwarz only
on the independent environment copy. The nonlocal term in
$(I-aL_1)p$ gives the necessary additional $ad_P$ in its
pointwise bound. RFK retains this term in its weighted coefficient
$b(1+a)X_2d_P$. Its unweighted coefficient uses the separate
exact $L^2$ contraction of the symmetric count operator, so no
pointwise contraction is asserted.

The actual precap identity is
$z=(m-aa_2(y))w-tx_1+aM_2(y)$ with
$x_1=mX_\theta+tU_\theta$. Its coefficient is at least
$A=m-a>0$. The native radial identity
$\|DC_V(z)z\|\le V/4$ therefore bounds the actual root product
$\|DC_V(z)w\|$ by $S_w(x_1)$. The pointwise bound
$|U_\theta|\le V_c$ supplies $S_w\le S_0+S_x|X_\theta|$;
the own-law count-energy contraction separately gives
$\|U_\theta\|_2\le r_0$ and hence $\|S_w\|_2\le\overline S$.
These two uses have the distinct correct scopes.

In $B_2$, its independent population environment copy permits
$\mathbb E'|R'||w'|\le d_YM_w$. The local root product
$S_w|R|$ remains inside its norm until it is bounded by
$S_0d_Y+S_xH_Y$ using the proved source polynomial. This yields
the stated radial consumer without factoring the actual cap Jacobian
from a correlated displacement. Intermediate own population laws
have $\|P_\theta\|_2\le.55$ by Minkowski and
$\|U_\theta\|_2\le.55$ by their symmetric count operator.
Thus their full OU second moment is bounded by
$c^2(.55+tX_2)^2+dq^2<.70^2$, as required.

For random finite arrays, the environment moment would be a random
empirical quantity correlated with the remaining factors. The source
does not replace it by $.70$ and explicitly confines (RFK.14) to
the population carrier. Its finite statement remains the mixed
conditional estimate in the preceding lemma.
:::

(sec-rfsa-principal)=
## 4. Principal and signed algebra, with exact rational checks

:::{prf:remark} Principal and signed balance audit
:label: rem-rfsa-principal

The principal theorem's operators are self-adjoint on the same
normalized finite-array Hilbert space, or on the lifted coupled-root
population space. Count symmetry and $K\le1$ give $0\le L_j\le I$,
including for the actual noisy second graph. The native cap Jacobian
is a positive self-adjoint multiplication operator bounded by $I$.
No commutation between the two count operators is used. The full
signed differential (RFK.5)--(RFK.6) then includes the spatial
derivative forces exactly; the contraction of its principal part
does not erase those forces.

Independent exact rational checks using the source's terminating
interval bounds give the following strict certificates. The two
diagonal entries of the majorant deficit after subtracting
$\operatorname{diag}(.0015,.073)$ exceed $.00006$ and $.0008$;
its off-diagonal absolute value is below $.0001$. The resulting
determinant lower bound is $3.8\times10^{-8}>0$.
At $a=.006$, the two count-alignment absorption coefficients
strictly exceed $.00149$ and $.0721$. The source's radial-consumer
rational upper calculations give coefficients strictly below
$.0093981$ and $.000365531$, hence below the displayed
$.0094$ and $.000366$.

The pair forms (RFK.7)--(RFK.9) use the same conductances as the
retained negative alignment forms. The independent fresh-OU bound
is applied after the pointwise kernel inequality; the jitter polynomial
is conditioned on its complete coupled source plan. These orderings
are exactly what permit the mixed-product conclusions.

Audit endpoint: all stated intermediate estimates in the frozen source
pass this second review, including their population/finite distinctions.
The actual full signed absorption, changing rooted preparation and
marked/dead feedback remain separate unclosed consumers. This audit
does not assert a default full active convergence or QSD mixing theorem.
:::

(sec-rfsa-zero-displacement)=
## 5. Zero-displacement precision update

:::{prf:remark} Preserved audit history and the non-strict norm product
:label: rem-rfsa-zero-displacement

The original second review above was performed on source SHA-256
`599f7f1bc4a56d7fc8dbb57343090e612a3a19ff1aa169557b645c1ec44a52b3`.
The subsequently frozen source SHA-256 is
`ae857ee67f7c1d44bbc2b3dd88cf9bf61ac3e1fdcae0d00dbe6b244dd25ac207`.
Its only proof-expression change replaces the final strict product
bound in (RFK.14) by
$$
a\|DC_V(z_\theta)B_2\|_2\le.0094d_X+.000366d_P.
$$
This includes $d_X=d_P=0$, where both sides vanish. The strict
scalar coefficient margins audited in Section 4 remain unchanged;
the proof establishes this non-strict product bound for every
displacement. The corrected statement passes the same second audit.
No consumer hypothesis or population/finite scope has changed.
:::
