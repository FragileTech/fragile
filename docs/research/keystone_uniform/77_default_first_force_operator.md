# Centered operator bound for the actual first count force

(sec-ffo77-register)=
## 1. Actual general first-force register

:::{prf:definition} Prepared-law first spatial derivative operator
:label: def-ffo77-register

Retain the unchanged harmonic count update and interpolation of
{prf:ref}`def-dbl68-register`. At a fixed interpolation point let
$(X,P)$ be the actual prepared root law on a probability space
$(\Omega,\mu)$, and let $r=\dot X$ be any square-integrable vector
displacement. The velocities obey the actual preparation bound
$|P|\le V_c=4$. The positions may be arbitrary measurable coordinates:
the operator estimate below uses no compact support, source-tail cutoff
or position moment. Define
$$
K(D)=e^{-|D|^2/2},\qquad G(D)=K(D)D,\qquad
\ell=\sup_D|G(D)|=e^{-1/2}.
$$
For an independent copy of the complete root put $D=X-X'$ and
$H=P-P'$. The complete own first spatial force is the linear operator
$$
(\mathcal B_Pr)(\omega)
 =\mathbb E'[H\,G(D)^{\mathsf T}(r-r')].
\tag{FFO.1}
$$
It equals $B_1=-L_{\dot k_1}P$ because
$\dot k_1=-K(D)D\cdot(r-r')$.

The root law includes its actual measurement, accepted source/component
plan, original-slot Haar velocities and all original recipient jitters.
No independence between $r,P$ or $X$ is assumed. An independent
environment is an independent copy of that entire joint law.

For a finite array use the uniform probability space
$\Omega=\{1,\ldots,N\}$, its actual normalized averages, and the
exact denominator $N$ in (FFO.1). Self forces are zero. For a
random finite array first condition on its entire prepared array and
physical displacement, then take its actual outer expectation.

Write
$$
m_P=\mathbb E|P|,\qquad
f^\circ=f-\mathbb Ef,\qquad
\operatorname{Var}(f)=\|f^\circ\|_2^2.
\tag{FFO.2}
$$
The matrix $HG(D)^{\mathsf T}$ need not be symmetric.
Neither self-adjointness of $\mathcal B_P$ nor a favorable sign
of its quadratic form is presumed.
:::

(sec-ffo77-pair)=
## 2. Pair reciprocity and the complete weighted energy

:::{prf:lemma} Exact antisymmetric first-force pairing
:label: lem-ffo77-pair

The operator (FFO.1) kills constant displacements, has zero-mean
output, and obeys the exact identity
$$
\langle s,\mathcal B_Pr\rangle
 =\frac12\mathbb E_{\mathrm{pair}}
    [((s-s')\cdot(P-P'))\,
                       (G(X-X')\cdot(r-r'))]
\tag{FFO.3}
$$
for arbitrary square-integrable test vectors $s,r$. Consequently
$$
|\langle s,\mathcal B_Pr\rangle|
 \le \ell\,\sqrt{\mathcal E_P(s)\mathcal E_P(r)},
$$
$$
\mathcal E_P(f)=\frac12\mathbb E_{\mathrm{pair}}
   [ (|P|+|P'|)|f-f'|^2].
\tag{FFO.4}
$$
These are exact population identities and inequalities, and also exact
normalized finite-array identities and inequalities.
:::

:::{prf:proof}
The force integrand in (FFO.1) changes sign under interchange
of its two roots. Indeed $H$ and $G(D)$ each change sign,
whereas $G(D)\cdot(r-r')$ remains unchanged.
Integrating the integrand over both roots gives zero.
A constant $r$ gives $r-r'=0$, so constants are killed.

Now interchange the two roots in $\langle s,\mathcal B_Pr\rangle$
and average the original and interchanged expressions.
This gives (FFO.3), with no matrix-transpose assertion.
The integrability follows directly from
$|H|\le2V_c$, $|G(D)|\le\ell$ and Cauchy--Schwarz
for the two square-integrable displacement differences.

Bound the absolute integrand by
$$
\ell(|P|+|P'|)|s-s'||r-r'|.
$$
Weighted Cauchy--Schwarz on the pair probability measure gives
(FFO.4), including the two factors $1/2$.
All operations are the same exact sums with factor $N^{-2}$
for finite arrays. The self terms vanish from the differences;
including them changes neither side.
:::

:::{prf:lemma} Exact centered additive-edge energy
:label: lem-ffo77-weighted-energy

For every square-integrable $f$,
$$
\mathcal E_P(f)
 =\mathbb E[|P||f^\circ|^2]
                 +m_P\operatorname{Var}(f)
 \le (V_c+m_P)\operatorname{Var}(f).
\tag{FFO.5}
$$
The local product $|P||f^\circ|^2$ is retained in the
identity. Its upper bound uses the individual preparation speed,
not a product of its two averaged factors.
:::

:::{prf:proof}
Replace $f$ by $f^\circ$ in its pair differences. Symmetry
of the additive pair weight gives
$$
\mathcal E_P(f)=\mathbb E_{\mathrm{pair}}
              [|P||f^\circ-f^{\circ\prime}|^2].
$$
Expand this square. The first term is
$\mathbb E[|P||f^\circ|^2]$. The second is
$m_P\mathbb E|f^\circ|^2$ by independent roots.
The cross term is
$$
-2\,\mathbb E[|P|f^\circ]\cdot\mathbb E f^{\circ\prime}=0.
$$
This is a product across the independent environment copies;
it does not factor the possibly correlated local variables
$|P|$ and $|f^\circ|^2$. Their pointwise bound
$|P|\le V_c$ proves the last inequality in (FFO.5).
The argument works verbatim for uniform-index finite sums.
:::

(sec-ffo77-norm)=
## 3. General operator norm below the positional threshold

:::{prf:theorem} Centered first-force norm for arbitrary prepared shapes
:label: thm-ffo77-operator

Under {prf:ref}`def-ffo77-register`,
$$
\|\mathcal B_Pr\|_2
 \le \ell(V_c+m_P)\|r^\circ\|_2.
\tag{FFO.6}
$$
If the population prepared velocity RMS is at most $r_0$, then
$$
\|\mathcal B_Pr\|_2
 \le \ell(V_c+r_0)\|r^\circ\|_2.
\tag{FFO.7}
$$
At the proved default post-burn population budget $r_0=.55$,
the coefficient is strictly less than $2.76185$.
A fixed finite prepared array with RMS at most $.56$
has coefficient strictly less than $2.76792$.

For a random prepared finite array, with its actual
$m_N=N^{-1}\sum_i|P_i|$ and
$\operatorname{Var}_N(r)=N^{-1}\sum_i|r_i-\bar r|^2$,
the full assertion is the mixed expectation
$$
\mathbb E\langle|\mathcal B_Pr|^2\rangle_N
 \le\ell^2\,\mathbb E[
                  (V_c+m_N)^2\operatorname{Var}_N(r)].
\tag{FFO.8}
$$
Replacing $m_N$ by its actual RMS $r_P$ inside this
expectation is also valid. Replacing that mixed expectation by a
product of an averaged velocity coefficient and an averaged
displacement is not asserted.
:::

:::{prf:proof}
By (FFO.3)--(FFO.5),
$$
|\langle s,\mathcal B_Pr\rangle|
 \le\ell(V_c+m_P)\|s^\circ\|_2\|r^\circ\|_2.
$$
The force has zero mean. Hilbert-space duality, or choosing
$s=\mathcal B_Pr$ in the last inequality, gives (FFO.6).
The zero-force case is covered directly, so no division by a
possibly zero norm is required. Cauchy--Schwarz gives
$m_P\le\|P\|_2\le r_0$, proving (FFO.7).

The elementary exponential series gives
$e>1+1+1/2+1/6+1/24+1/120=163/60$.
The exact rational comparison
$(607/1000)^2(163/60)>1$ proves $\ell<.607$.
Thus
$$
\ell(4+.55)<.607(4.55)=2.76185,\qquad
\ell(4+.56)<.607(4.56)=2.76792.
\tag{FFO.9}
$$
For a random finite array apply (FFO.6) conditional on the
entire prepared array and displacement, square, and integrate.
This proves (FFO.8) without conditioning any future noise
on an empirical event. The pointwise empirical inequality
$m_N\le r_P$ gives the stated alternative.
:::

(sec-ffo77-position)=
## 4. Actual first-provider positional feedback passes at the default

:::{prf:corollary} Complete position differential and its centered means
:label: cor-ffo77-position

At the unchanged harmonic step put
$$
t=.02,\qquad \nu=.3,\qquad a=t\nu=.006,\qquad
c=e^{-.04},\qquad b=t(1+c),\qquad a_x=1-tb.
$$
Let $p=\dot P$, $L_X$ be the actual own first count
Laplacian and
$$
E=\dot U=(I-aL_X)p+a\mathcal B_Pr.
$$
The actual shared-noise final position differential is
$$
R=\dot x^+=a_xr+bE.
\tag{FFO.10}
$$
It has exact mean and centered bounds
$$
\mathbb ER=a_x\mathbb Er+b\mathbb Ep,\qquad
\|R^\circ\|_2
 \le \rho_x\|r^\circ\|_2+b\|p^\circ\|_2,
$$
$$
\rho_x=a_x+ab\ell(V_c+m_P)
 =1-tb[1-\nu\ell(V_c+m_P)].
\tag{FFO.11}
$$
Hence the population post-burn budget $V_c=4,r_0=.55$
gives $\rho_x<.999866<1$.
A fixed finite prepared array with $r_P\le.56$ gives
$\rho_x<.999868<1$.
Both rates use every actual first conductance and the exact finite
count denominator when applicable. They do not charge a Gaussian
tail or a particle floor.

The complete norm also satisfies
$$
\|R\|_2\le \rho_x\|r\|_2+b\|p\|_2.
\tag{FFO.12}
$$
Thus the own first-provider spatial feedback alone passes the
exact positional absorption threshold
$\ell(V_c+m_P)<1/\nu=10/3$.
This positional differential estimate does not establish phase
contraction, preparation contraction, terminal-alive transport or
an iterated default mixing rate.
:::

:::{prf:proof}
The exact first-count derivative gives $E$.
The unchanged harmonic/OU stages give
$y=a_xX+bU+tq\xi$ and the final position is
$x^+=y+s\zeta$. The original OU and final innovations are
shared independently of the input coupling, so their
differentials are zero. The second count kick and the native cap
act only on the velocity; they do not change this position formula.
This proves (FFO.10).

The own count Laplacian is self-adjoint, $0\le L_X\le I$,
kills constants and has zero-mean output. These properties follow
from its exact symmetric pair form with $0\le K\le1$ and
the original normalized denominator. Since $0\le a\le1$,
$\|(I-aL_X)p^\circ\|_2\le\|p^\circ\|_2$.
Use also $\mathbb E\mathcal B_Pr=0$ and (FFO.6) in
(FFO.10). This gives the mean and centered inequality in
(FFO.11). The full norm follows from the same argument,
$\|(I-aL_X)p\|_2\le\|p\|_2$ and
$\|r^\circ\|_2\le\|r\|_2$.

For the strict coefficients use $.96<c<.9608$, hence
$b>.0392$. The population coefficient in (FFO.9) gives
$$
1-\nu\ell(V_c+m_P)>.171445,\qquad
tb[1-\nu\ell(V_c+m_P)]
 >.02(.0392)(.171445)=.00013441288.
$$
Thus $\rho_x<.99986558712<.999866$.
The fixed finite coefficient gives instead
$$
1-\nu\ell(V_c+m_P)>.169624,\qquad
tb[1-\nu\ell(V_c+m_P)]
 >.02(.0392)(.169624)=.000132985216,
$$
which proves $\rho_x<.999867014784<.999868$.
Every scalar comparison is between exact terminating rationals.

Finally $1-a_x=tb$ implies
$(1-a_x)/(ab)=t/a=1/\nu$ exactly.
This identifies the first-feedback threshold without a rounded
approximation. Only this positional feedback has been absorbed.
:::

:::{prf:corollary} Endpoint positional estimate for a valid prepared coupling
:label: cor-ffo77-position-endpoints

Couple two actual prepared population laws with bounded velocities
$|P_j|\le4$ and $\|P_j\|_2\le.55$. Let their coupled
displacements be $r=X_1-X_0$ and $p=P_1-P_0$.
Share their complete original kinetic innovations independently of
this input coupling. Their final physical positions obey
$$
\|x_1^+-x_0^+\|_2
 \le .999866\,\|X_1-X_0\|_2
                 +b\,\|P_1-P_0\|_2.
\tag{FFO.13}
$$
The same estimate holds for fixed finite arrays with coefficient
$.999868$ if both endpoint empirical prepared RMS values are
at most $.56$, retaining their original row matching.

In particular a valid coupling with $P_1=P_0$ has strict
positional contraction under the actual own first provider,
including unrestricted source shape changes. The statement is a
physical positional estimate before terminal restriction; it
does not identify a coupling of separately surviving alive
empirical measures.
:::

:::{prf:proof}
Interpolate the coupled prepared endpoints. Their intermediate
velocities remain individually bounded by four by convexity.
Minkowski gives intermediate RMS at most the respective
endpoint budget. Equations (FFO.10)--(FFO.12) therefore hold
uniformly along the interpolation with the displayed coefficient.
The operator kernels and their derivatives are bounded;
domination by the square-integrable input displacements justifies
differentiation and integration along the path. Shared Gaussian
innovations retain each own transition marginal throughout.
Integrate the position differential and apply Minkowski to obtain
(FFO.13). The fixed-array proof is the same normalized
finite-dimensional argument.
:::

(sec-ffo77-signed)=
## 5. Signed consumer and precise remaining scope

:::{prf:remark} Weighted bilinear interface for the complete cap account
:label: rem-ffo77-signed

Equation (FFO.4) is stronger than using the scalar norm alone:
it retains the plan-local weighted energy
$\mathbb E[|P||f^\circ|^2]$ inside its exact expression.
For the oriented first response of research68 it gives
$$
\begin{split}
2\langle A_rr+A_pp,\mathcal B_Pr\rangle
\le{}&2\ell
 \sqrt{\mathcal E_P(A_rr+A_pp)\mathcal E_P(r)},\\
2\langle A_rr+A_pp,F_1\rangle
={}&2\langle A_rr+A_pp,\mathcal B_Pr\rangle
                 -2\langle A_rr+A_pp,L_Xp\rangle.
\end{split}
\tag{FFO.14}
$$
The exact second term has not been assigned a favorable sign:
it retains both alignment and the position--velocity cross form.
Likewise the first-provider force square, the correlated second
provider, and the cap-force terms in research74 remain in the
complete physical account. A positional coefficient below one
does not absorb those terms or contract the actual full phase law.

The present improvement comes from exact pair antisymmetry and
the centered additive-edge energy. It does not infer a small
pointwise speed from the population RMS or factor a local
source/velocity product. It applies to the actual first provider
for arbitrary prepared shapes, including the original unbounded
recipient jitters. Source73 remains unchanged and provides its
separate exact Gaussian pair/triple consumer.

Finite random-budget exceptions and future restrictions still
require their actual mixed displacement charges. Nothing here
replaces the current-survival law by fresh postconditioning
Gaussians, drops either swarm's own normalization, proves
preparation or mark feedback, or identifies a finite QSD.
:::

(sec-ffo77-checks)=
## 6. Exact coefficient certificate and preserved inputs

:::{prf:remark} Rational checks and frozen dependencies
:label: rem-ffo77-checks

The coefficient comparisons are checked by this exact certificate.

```python
from fractions import Fraction as F

assert F(607, 1000) ** 2 * F(163, 60) > 1
population_force = F(".607") * (F(4) + F(".55"))
finite_force = F(".607") * (F(4) + F(".56"))
assert population_force == F("2.76185") < F(10, 3)
assert finite_force == F("2.76792") < F(10, 3)

population_gap = F(".02") * F(".0392") * (
    F(1) - F(".3") * population_force
)
finite_gap = F(".02") * F(".0392") * (
    F(1) - F(".3") * finite_force
)
assert population_gap == F(".00013441288")
assert finite_gap == F(".000132985216")
assert F(1) - population_gap < F(".999866")
assert F(1) - finite_gap < F(".999868")
print("All centered first-force rational comparisons pass.")
```

The input records are retained at the following SHA-256 values.

| Record | SHA-256 |
|---|---|
| 69 | `012f6f1f2671ddac98e10b4d63abca7662a5759deeb8267c475a399425a01a10` |
| 73 | `3d8338a7cd399f9d746ae6b9bac8e8f847fdb2ba33cec83be235ba46e022a67a` |

The proof of (FFO.6) is analytic pair symmetrization and
weighted Hilbert-space duality. No numerical operator eigenvalue
is used to certify its validity.
:::
