# Independent audit of the default nonconstant-velocity cap estimate

(sec-csba-source)=
## 1. Exact reviewed source and endpoint

:::{prf:definition} Reviewed cap-shape source
:label: def-csba-source

This independent mathematical review reads research source
`60_default_cap_shape_bands.md` at SHA-256
`3666f9c63026ebfcad241005b5b1926549707f238dde56412818364cb4629a82`.
It reviews the complete actual harmonic count kinetic update at
$h=.04$, $\nu=.3$, including both own count fields, the correlated
OU second provider, original final Gaussian and native cap.

The endpoint is the one-update optimal physical law estimate
$$
W_{2,G}(K_{\rm ph}\lambda_0,K_{\rm ph}\lambda_1)^2
\le\frac{1039}{1040}W_{2,G}(\lambda_0,\lambda_1)^2,
$$
and its exact finite-array version with the normalized array metric,
provided each prepared law has all velocities within $1/200$
of its own deterministic center of norm at most $2$.
The two centers can differ. Positions need only be square-integrable.
Terminal marks and conditional-alive normalization are excluded
from this physical metric, as explicitly stated in the source.
The review does not turn the one-update class into an invariant
or a population attraction theorem.
:::

(sec-csba-forces)=
## 2. Complete conditional force budgets

:::{prf:lemma} Independent verification of the pair and graph estimates
:label: lem-csba-pair-review

The estimates (CSB.2)--(CSB.4) are valid for both actual population
providers and every exact finite-array kinetic kernel, uniformly
in array size. In particular their pair coefficient is strictly
less than $.18$, and the complete spatial forces satisfy
$$
\|B_1\|_2\le.009d_X,\qquad
\|B_2\|_2\le.6\|R\|_2.
$$
:::

:::{prf:proof}
Along any coupling interpolation, every within-law velocity lies
within $\delta_v=1/200$ of the interpolated center. Hence every
pair velocity difference has norm at most $2\delta_v$. The
first count average is convex at $a=.006$, so this bound also
holds for its actual first outputs $U,U'$.

Freeze the whole prepared pair, before the fresh independent
OU draws. Then $D=X-X'$, $H=U-U'$ and the complete position
differential $R$ are fixed. The actual pair
$$
Y=a_xD+bH+tq\Xi,\qquad W=c(H-tD)+q\Xi,\qquad
\Xi\sim N(0,2I_3)
$$
satisfies exactly
$W=(c/a_x)H-(ct/a_x)Y+(qm/a_x)\Xi$.
Young's inequality with $1/50$, followed by
$\sup r^4e^{-r^2}=4/e^2$,
$\sup r^2e^{-r^2}=1/e$ and
$\mathbb E|\Xi|^2=6$, gives exactly (CSB.3).
This argument never factors the correlated $Y,W$.
Its terminating-rational upper bound is
$$
\frac{51}{50}\left[
\frac{8(.0004)(.136)}{.998}
+\frac{12(.0392)(.368)}{.998}\right]
+\frac{51(.0001)(.368)}{.998}
=.179248545090180\ldots<.18.
$$
Only the exact rational inequality is used; the displayed decimal
is a diagnostic rendering.

The first kernel gradient has norm at most $\ell=e^{-1/2}$.
Jensen gives
$\|B_1\|_2^2\le8\delta_v^2\ell^2d_X^2$,
whose coefficient square root is less than $.009$.
The second Jensen estimate is taken before OU integration:
$$
\|B_2\|_2^2\le
\mathbb E[|R-R'|^2|Y|^2e^{-|Y|^2}|W|^2]
\le C_v\mathbb E|R-R'|^2
\le2C_v\|R\|_2^2.
$$
Crucially $R$ here includes $aB_1$ and is fixed before this
Gaussian pair draw. The estimate is not applied to its principal
part alone.

For arrays, rowwise Jensen uses the actual denominator $N$,
with total weights at most one and exact zero self term.
Every distinct pair has the stated Gaussian noise difference.
The exact identity
$$
\frac1{N^2}\sum_{i,j}|R_i-R_j|^2
=2\left[\frac1N\sum_i|R_i|^2
-\left|\frac1N\sum_iR_i\right|^2\right]
$$
completes the same conditional bound. Outer integration retains
all array dependence, including the first graph and prepared
source plan. No independent second-graph rows, unconditional
moment factorization or random-to-population substitution is used.
For $N=1$ both graph forces are exactly zero, so the estimates
remain valid without a pair draw.
:::

(sec-csba-cap)=
## 3. Cap orientation, operator sectors and exact arithmetic

:::{prf:lemma} Independent verification of full cap-square absorption
:label: lem-csba-absorption-review

The complete derivative estimate (CSB.5) is valid:
$$
\mathbb E Q_\beta(\dot x^+,\dot v^+)
\le Q-.001d_X^2-.04d_P^2.
$$
It restores both full graph forces before the final cap square.
:::

:::{prf:proof}
Both count operators are self-adjoint on the actual normalized
array or lifted population space and have spectra in $[1-a,1]$.
The principal proof in research 38 directly bounds
$\|R_0\|_2^2+\|T_0\|_2^2$, not merely the smaller capped
quadratic. Its loss is at least $.00149d_X^2+.0721d_P^2$.
The argument separately applies each alignment spectral inequality;
it does not commute the two realized operators.

The exact oriented expression is
$$
T_0=[fI+ctaL_2]r+[cA_2+(\beta-t)bI]A_1p,
\quad f=(\beta-t)a_x-ct.
$$
The retained intervals give $0<f+cta<.0009$ and
$c+(\beta-t)b<.962$. Thus the source's oriented principal
bounds for $R_0,T_0$ are valid even though $p$ is nonconstant.
Its full-force budgets have exact terminating-rational checks
$$
ba(.009)<.00000212=\alpha,
$$
$$
a(.962)(.009)+.0036(.999216+\alpha)
=.003649133232<.00365=e_X,
$$
$$
.0036(.039216)=.0001411776<.000142=e_P.
$$
These inequalities integrate the complete $R=R_0+baB_1$.

For the actual native-cap Jacobian $D_C$, $0\le D_C\le I$.
Writing $e=Z-D_CZ$ gives
$$
|Z|^2-|D_CZ|^2+2\beta\langle R,e\rangle+\beta^2|R|^2
=|e+\beta R|^2+2\langle D_CZ,e\rangle\ge0.
$$
Hence the source's bound
$Q_\beta(R,D_CZ)\le|R|^2+|Z+\beta R|^2$
holds pointwise with its actual correlated cap argument.
No standalone cap contraction in the cross metric is assumed.

Restore the full $E_R,E_T$ in those two squares. Their exact
additional coefficients are the source's $A_X,A_P,A_{XP}$.
With Young parameter $.0004$, the exact rational differences are
$$
.00149-A_X-.0004=.0010658708196656>.001,
$$
$$
.0721-A_P-\frac{A_{XP}^2}{.0004}
=.040990898415987\ldots>.0409>.04.
$$
The latter decimal is again only a rendering of an exact positive
Fraction comparison. The inequalities prove the claimed derivative
bound, including zero displacement. All final norm bounds are
non-strict, while only the numerical coefficient margins are strict.
:::

(sec-csba-law)=
## 4. Marginals, optimal transport and actual class examples

:::{prf:theorem} Law endpoint and scope audit
:label: thm-csba-law-review

The physical optimal law estimate (CSB.8), the original-component
readout example, and the source's noninvariance limitation all pass
this independent audit.
:::

:::{prf:proof}
Every coupling of the two prepared laws preserves both pointwise
velocity-band marginal conditions. Their convex interpolation
therefore has the same band radius about its interpolated center.
Share only the subsequent original kinetic Gaussians. In every
marginal those noises have exactly their independent original laws;
the two own count providers are their own actual joint stage laws.
The full derivative bound holds along this interpolation.
Pointwise provider differentiation is justified by bounded first
kernel derivatives, bounded prepared velocities and the
square-integrable positions. The second-stage derivative has the
uniform conditional pair bound proved above, yielding an $L^2$
derivative uniformly over interpolation. These bounds justify its
integration; alternatively truncate the square-integrable inputs
and pass the same uniform bound to the limit.

Since $Q\le1.04(d_X^2+d_P^2)$,
the smaller derivative loss $.001$ gives factor
$1-.001/1.04=1039/1040$.
Integrate the derivative and apply Jensen to the positive
quadratic $G$. The result is a valid coupling of the actual
physical output marginals. Taking the infimum over all prepared
input couplings therefore gives the correctly directed optimal
Wasserstein inequality. The finite result applies the same argument
on the full array space, with normalized cost, not on an assumed
product of independent output rows.

If every original frozen velocity lies within $1/400$ of $u$,
each original component mean is also within that radius.
The original Haar formula then bounds
$$
|P_i-u|\le|\bar v_C-u|
+|\alpha_{\rm col}||v_i-\bar v_C|
\le(1+2|\alpha_{\rm col}|)/400=1/200
$$
for every component and Haar outcome. Positional copying,
mandatory revival and jitter do not copy original velocities.
Thus this readout is exact, rather than a donor-velocity model.

The source's nonconstant population example is well-defined:
two interior types with distinct original velocities can be put
in any sufficiently small phase ball. Positive standardizer
floors and smooth bounded comparison features make the entire
fitness oscillation at most a finite constant times its diameter.
At any fixed positive powers this supplies a positive radius
with alive incoming acceptance column at most $1/8$, giving the
declared finite rooted components. Cross-type measurements have
the same symmetric diversity distance but unequal radial rewards,
so the increasing positive reward factor gives a strictly positive
accepted copying event. Each type also has positive isolation
probability and retains its distinct velocity on that event.
This verifies actual nonconstant prepared velocities and genuine
active copying within the stated class.

Finally the band is not asserted to be invariant. At $N=1$
and zero input, actual count forces vanish and the precap velocity
is $mq\xi$. Because $mq>0$, the native capped velocity has
the full open radius-$V$ ball as support. No radius-$1/200$
band can contain this law. Thus a known RMS burn cannot license
iteration of the band estimate. Marks, alive normalization,
preparation cost and full nonlinear attraction remain separate
as the reviewed source explicitly states.
:::

(sec-csba-result)=
## 5. Review result

:::{prf:remark} Accepted source revision
:label: rem-csba-result

The frozen source specified in Section 1 passes the independent
force, cap, operator, marginal, transport and scope checks.
Exact rational endpoint checks pass. No correction to source 60
is required. The accepted endpoint is the complete one-update
physical law contraction on the explicit nonconstant prepared
velocity bands, with no population-size dependent floor.
This review supplies no default general-population, conditional-alive
or multistep convergence conclusion.
:::
