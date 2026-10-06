# Independent audit of conditional cap loss and the general first force

(sec-ccra-source)=
## 1. Frozen reviewed source and actual scope

:::{prf:definition} Conditional cap review record
:label: def-ccra-source

This review reads
`69_default_conditional_cap_loss.md` at SHA-256
`012f6f1f2671ddac98e10b4d63abca7662a5759deeb8267c475a399425a01a10`.
It includes the appended first-force lemma (CCL.12)--(CCL.14).
The carrier is the complete actual harmonic count kinetic
differential at $h=.04$, $\nu=.3$, $V=2$, with both own
providers, original Gaussian innovations, native cap and exact
count normalizers. All source/component/jitter correlations
are permitted through its actual prepared input.

The endpoint is a weighted conditional cap-sector loss and a
sharper general first-force bound. It is not a positive
completed multi-step margin, an alive-law transfer or an
invariant comparison class.
:::

(sec-ccra-square)=
## 2. Pointwise completion keeps the entire velocity differential

:::{prf:lemma} Exact cap-square audit
:label: lem-ccra-square

The identities and inequalities (CCL.3)--(CCL.5) are correct
pointwise for the actual cap Jacobian and complete correlated
velocity differential.
:::

:::{prf:proof}
Write $D=DC_V(z)$ and
$A=\beta(I+D)^{-1}R$. The native derivative has radial
eigenvalue $d_r=[V/(V+|z|)]^2$ and tangential eigenvalue
$d_t=V/(V+|z|)$, with $0<d_r\le d_t\le1$.
All factors depending only on $D$ commute.
The actual cap-loss expansion is
$$
\mathcal L_C
=Z^{\mathsf T}(I-D^2)Z+
2\beta R^{\mathsf T}(I-D)Z+\beta^2|R|^2.
$$
Since $(I-D^2)(I+D)^{-1}=I-D$, the completed square
$(Z+A)^{\mathsf T}(I-D^2)(Z+A)$ gives the exact cross term.
Its position-square contribution plus the stated residual is
$$
\beta^2R^{\mathsf T}(I-D)(I+D)^{-1}R
+2\beta^2R^{\mathsf T}D(I+D)^{-1}R
=\beta^2|R|^2.
$$
This proves (CCL.3), including $z=0$ where $D=I$.
No inverse of $I-D$ or cap/graph commutation is used.

The increasing function $d/(1+d)$ gives
$$
2\beta^2D(I+D)^{-1}\ge
\frac{2\beta^2}{1+(1+|z|/V)^2}I.
$$
Also $I-D^2\ge(1-d_t^2)I$.
These prove the two lower bounds, retaining the complete
correlated force-centered velocity square. There is no
independent replacement of its $Z$.
:::

(sec-ccra-conditioning)=
## 3. Actual pre-OU conditional expectation and full-jitter weight

:::{prf:lemma} Conditional coercivity audit with finite own providers
:label: lem-ccra-conditioning

The complete conditional bounds (CCL.6)--(CCL.11) hold in
their declared population and finite-array forms, retaining
all original Gaussian outcomes and actual prepared dependence.
:::

:::{prf:proof}
The exact relation $y=x_1+tw$ holds with
$x_1=mX+tU$ and $m=1-t^2$.
For the actual joint second provider,
$L_yw=d_yw-M_y$, giving
$z=(m-ad_y)w-tx_1+aM_y$.
For arrays the count averages may include their actual self
numerator and denominator terms: they cancel exactly in
$L_yw$. The resulting $0\le d_y\le1$ and
$m-a=.9936>0$ justify the pointwise upper bound
$$
|z|\le m|w|+t|x_1|+a\,\text{actual provider average of }|w'|.
$$
No independence of that field and its own OU is required.

Conditioning on the full actual pre-OU information fixes the
root $X,U,R$ and, in arrays, the entire prepared array and
physical differential. Thus
$$
\mathbb E[|z|\mid\mathscr P]\le
(mc+t^2)|U|+mb|X|+mqg_d
+a[c(\bar U_1+t\bar X_1)+qg_d].
$$
Here $mb=r_H$. For finite arrays the provider first moment
is averaged over the actual fresh row noises by linearity
conditional on this entire preparation. It is not replaced
by a population quantity. This proves precisely the source's
$C_*(X,U)$.

The function
$f(u)=2\beta^2/[1+(1+u/V)^2]$
is decreasing. Setting $v=1+u/V\ge1$ gives
$$
f''(u)=\frac{4\beta^2(3v^2-1)}
 {V^2(1+v^2)^3}>0.
$$
Hence conditional Jensen is in the correct direction:
$$
\mathbb E[\mathcal L_C\mid\mathscr P]
\ge |R|^2\mathbb E[f(|z|)\mid\mathscr P]
\ge |R|^2 f(\mathbb E[|z|\mid\mathscr P])
\ge |R|^2f(C_*).
$$
The multiplication by $|R|^2$ is legitimate because the
complete first-stage position differential is fixed before
fresh OU. In contrast the complete $Z$ and cap Jacobian
remain correlated and are not so factored.

For the actual source representation $X=S+I\sigma_JG$,
$|S|\le\sqrt dL$ and the original component readout gives
$|U|\le V_c$. In the population its true provider first
moment is bounded by $\sqrt dL+\sigma_Jg_d$.
This bounds $C_*$ by $C_0+C_J|G|$ exactly as in (CCL.10).
Monotonicity of $f$ gives (CCL.11), with its source
displacement factor still inside the weighted expectation.
The finite provider instead keeps
$\sqrt dL+\sigma_JN^{-1}\sum_j|G_j|$ inside that weight.
No product with a correlated $R_i$ is replaced by its
unconditional jitter mean. The source representation is
an explicit hypothesis whenever this corollary is used
at a comparison interpolation; its preservation along
an arbitrary coupling path is not asserted here.
:::

(sec-ccra-first)=
## 4. The sharper first-force bound keeps the local product

:::{prf:lemma} Independent-environment first-force audit
:label: lem-ccra-first

The source's bound
$$
\|B_1\|_2\le\ell(V_c+3r_0)d_X
$$
is correct for a deterministic prepared population provider or
a fixed finite empirical array with its stated RMS bound.
The random finite-array conclusion correctly retains
$$
\ell^2\mathbb E[(V_c+3r_P)^2(d_X^{\rm arr})^2].
$$
:::

:::{prf:proof}
The exact first force is
$$
B_1=\mathbb E'[K(D)(D\cdot(r-r'))(P-P')],
\qquad D=X-X'.
$$
This sign matches $B_1=-L_{\dot k_1}P$ because
$\nabla K(D)=-DK(D)$.
Since $K(D)|D|\le\ell$, expanding the two triangle
factors gives the four source terms in (CCL.14).
The unprimed correlated local product $|r||P|$ is bounded
by $V_c|r|$. It is not assigned an RMS factor.
Inside the independent environment,
$$
\mathbb E'|P'|\le r_0,\quad
\mathbb E'|r'|\le d_X,\quad
\mathbb E'(|r'||P'|)\le r_0d_X
$$
by Cauchy--Schwarz, even though $r'$ and $P'$ can correlate.
Taking the root $L^2$ norm yields
$$
\ell[(V_c+r_0)d_X+d_X\|P\|_2+r_0d_X]
\le\ell(V_c+3r_0)d_X.
$$
For a fixed array the same Cauchy bounds are its actual
uniform-index averages, using its empirical RMS. Including
the zero self force in the triangle upper bound only
enlarges it and does not alter denominator $N$.
For a random array this is conditional on the complete
prepared realization. Squaring and integrating gives the
source's mixed expectation, rather than a product of
unconditional means.

The stated scalar endpoints are exact terminating rationals:
$$
.607(4+3(.55))=3.42955<3.43,\qquad
.607(4+3(.56))=3.44776<3.448.
$$
A finite moment-good restriction can use these endpoints
only on that event. Its complement, any correlated comparison
path and subsequent own survival require their separate
actual charges, as stated in the reviewed source.
:::

(sec-ccra-result)=
## 5. Review result and scope

:::{prf:remark} Accepted conditional coercivity revision
:label: rem-ccra-result

The frozen source specified in Section 1 passes this independent
cap-square, conditional-provider, full-Gaussian, source-weight
and first-force review. No source correction is required.
The existing cap body and appended general first-force
consumer are consistent and preserve their distinct hypotheses.

The accepted result does not certify that the remaining
signed first/second force forms are absorbed by the completed
loss. It also does not discharge preparation, terminal marks,
own alive normalization or chronological nonlinear feedback.
No uniform general law gap or default active convergence
endpoint is inferred from this audit.
:::
