# Independent audit of the signed first-provider/native-cap consumer

(sec-sfa84-input)=
## 1. Frozen input and verdict

:::{prf:proposition} Accepted signed auxiliary differential estimate
:label: prop-sfa84-verdict

The complete proof in `80_default_phase_signed_consumer.md` is accepted
at SHA-256
`bfa19ef3c4b8691646da36a86dd3b5966ac6e3e1ec14bd74b11c43b36d007957`.
Its population and fixed-finite-array conclusions are respectively

$$
\mathbb E Q_{.04}(R,DZ_b)
\le(1-.0000147)\mathbb E Q_{.04}(r,p),
$$

$$
\mathbb E_\xi\langle Q_{.04}(R,DZ_b)\rangle_N
\le(1-.0000048)\langle Q_{.04}(r,p)\rangle_N,
\tag{SFA84.1}
$$

under that record's exact prepared velocity and actual provider/finite
array budgets. Here $D$ is evaluated at the actual complete noisy precap
velocity, and $R,Z_b$ include the complete first count derivative.
The full second-field derivative $aDF_2$ is expressly absent from the
auxiliary pair in (SFA84.1) and expressly retained in its exact signed
remainder. No complete kinetic, transport, alive-law or repeated-time
claim follows from this accepted auxiliary estimate alone.

One numerical proof-line repair was required before acceptance. The
interval for $c$ has width $10^{-12}$, so the midpoint distance is less
than $5\cdot10^{-13}$. Multiplying by the derivative bound five gives
$2.5\cdot10^{-12}$, rather than $5\cdot10^{-13}$. The frozen record
now uses the conservative bound $5\cdot10^{-12}<10^{-9}$.
All multiplier entries, rates and exact certificates are unchanged.
:::

(sec-sfa84-matrix)=
## 2. Independent verification of the matrix certificate

:::{prf:lemma} Exact spectral polynomial and perturbation check
:label: lem-sfa84-matrix

Retain the exact rows $\mathsf R,\mathsf Z$, the matrices $G,Q_K,H$,
and positive $\tau$ in (SFC80.4). For either pair $(K,k)$ in
(SFC80.5), the spectral polynomial (SFC80.6) is positive semidefinite
for every $d\in[0,1]$ and the original $c=e^{-.04}$.
:::

:::{prf:proof}

**Step 1: check the multiplier independently.** Direct rational
determinants give the three leading principal minors of $H$ as

$$
\frac{137}{500000},\qquad
\frac{250840617}{10^{12}},\qquad
\frac{2862818517}{10^{20}}.
\tag{SFA84.2}
$$

They are strictly positive, so $H$ is positive definite. Its off-diagonal
entries cannot be individually discarded; the positive matrix as a whole
is the conditional cap-square consumer used below.

**Step 2: verify the spectral polynomials.** The audit independently
constructed the symbolic rational matrix

$$
(1-k)G-\mathsf R^{\mathsf T}\mathsf R
-d^2\mathsf Z^{\mathsf T}\mathsf Z
-\beta d(\mathsf R^{\mathsf T}\mathsf Z+
         \mathsf Z^{\mathsf T}\mathsf R)
-\tau Q_K+(d^2-.795)H-10^{-9}I
$$

at the declared rational midpoint of $c$. Its three leading principal
minor polynomials were obtained by independent symbolic determinants,
then expanded after $d=(j+u)/16$ for each $0\le j\le15$.
Their power coefficients were converted to Bernstein coefficients
by the identity

$$
u^i=\sum_{k=i}^n\frac{\binom{k}{i}}{\binom{n}{i}}
               \binom nk u^k(1-u)^{n-k}.
$$

All rational coefficients exceeded the strict floors stated in record80:

| $(K,k)$ | First minor | Second minor | Third minor |
|---|---:|---:|---:|
| $(2.76185,.00002)$ | $.0001495$ | $.0000360$ | $10^{-10}$ |
| $(2.76792,.00001)$ | $.0001543$ | $.0000373$ | $1.5\,10^{-10}$ |

The comparisons use exact rational arithmetic, not numerical eigenvalues.
Bernstein basis nonnegativity proves that every leading principal minor
is positive on every subinterval. Sylvester's criterion therefore gives
the original midpoint matrix at least $10^{-9}I$.

**Step 3: restore the exact exponential.** The degree-13/12 alternating
Taylor sums independently verify
$ .960789439152<c<.960789439153$. On $[.96,.961]$,
the exact affine rows and their derivatives have norms bounded by
$2,2,.03,1.1$, respectively. Their squared norms are convex quadratic
polynomials; checking their two rational endpoint values verifies those
four norm bounds on the full interval. Rank-one differentiation then
gives an operator derivative bound strictly below

$$
2(2)(.03)+2(2)(1.1)+2(.04)[(.03)(2)+(2)(1.1)]<5.
$$

The exact $c$ therefore changes the midpoint matrix by less than
$2.5\cdot10^{-12}<10^{-9}$. This absorbs the perturbation into the
proved midpoint reserve and proves the lemma without rounding the
algorithm's exponential coefficient.
:::

(sec-sfa84-noise)=
## 3. Actual joint-noise and first-field interfaces

:::{prf:lemma} The actual conditional square deficit consumes the multiplier
:label: lem-sfa84-conditional

The proof's use of the actual cap-square deficit is valid even though
$D$ depends on the root's OU vector, all finite OU rows and its own
correlated second count graph. It does not require $D$ to be scalar or
independent of the prepared first-force vectors.
:::

:::{prf:proof}

At each complete pre-OU preparation put $q=(r,e,f)$ with
$e=(I-aL_1)p$ and $f=B_1$. All three are fixed before the new OU
array. Diagonalizing the actual self-adjoint $D$ is permitted pointwise;
the scalar matrix polynomial is valid for every eigenvalue in $[0,1]$.
Recombining coordinates gives exactly

$$
\begin{split}
(1-k)Q_\beta(r,e)-Q_\beta(R,DZ_b)
\ge{}&\tau(K^2|r|^2-|f|^2)\\
&-\sum_{i,j}H_{ij}\langle q_i,(D^2-.795I)q_j\rangle.
\end{split}
\tag{SFA84.3}
$$

The random diagonalization did not commute $D$ with either count operator.
The identity is coordinate-independent. Since $H$ is positive definite,
the second line equals the negative sum of the quadratic forms
$\langle T_l,(D^2-.795I)T_l\rangle$, where
$T_l=\sum_i(H^{1/2})_{li}q_i$. These $T_l$ are fixed before OU,
so the actual conditional estimate of research74 applies to them.
Their conditional mean is nonpositive. This is a sum of conditional
fixed-vector inequalities, without factoring a correlated force from
an averaged cap matrix.

The first line need not be nonnegative pointwise. Its expectation is
nonnegative because the complete first-force operator in research77
obeys $\|f\|_2\le K\|r\|_2$. The population bound is taken in its
actual root probability measure; the finite bound is taken in its exact
normalized array measure, conditional on the complete prepared array.
The certificate multiplier $\tau$ is strictly positive. These two
independent quadratic consumptions therefore prove

$$
\mathbb E Q_\beta(R,DZ_b)\le(1-k)\mathbb E Q_\beta(r,e).
$$

Both own providers remain in the point of evaluation of $D$. All second
derivative terms remain in the separately declared $F_2$, including
$-L_2W$. The cap-square estimate has not been applied to that
OU-dependent vector. This precise separation is essential to the scope.
:::

:::{prf:lemma} The full first-count response has the stated finite cost
:label: lem-sfa84-first-count

The conversion from $(r,e)$ to the original input $(r,p)$ in record80
is valid with the declared population and finite final gaps.
:::

:::{prf:proof}

Since $0\preceq L_1\preceq I$, $e=p-aL_1p$ gives the exact expansion

$$
Q_\beta(r,e)-Q_\beta(r,p)
=-2a\langle p,L_1p\rangle+a^2\|L_1p\|^2
 -2\beta a\langle r,L_1p\rangle.
$$

Use $L_1^2\preceq L_1$ and complete the square in
$L_1^{1/2}p$ to obtain the upper bound
$\beta^2a\|r\|^2/(2-a)$. The possibly positive cross term has
not been assigned a favorable sign. The positive phase matrix satisfies
$Q_\beta\ge(1-\beta)(|r|^2+|p|^2)$, so its relative cost is at most

$$
C_1=\frac{\beta^2a}{(2-a)(1-\beta)}.
$$

The difference $(r,e)-(r,p)=(0,-aL_1p)$ has $Q_\beta$-norm at
most $a\|p\|$. The reverse triangle inequality in the full phase
Hilbert space therefore gives
$\mathbb E Q_\beta(r,e)\ge(1-a/\sqrt{1-\beta})^2
\mathbb E Q_\beta(r,p)\ge.9876\mathbb E Q_\beta(r,p)$.
The last scalar coefficient comparison is strict; the product estimate
is non-strict to include the zero input case. Exact rational arithmetic
verifies $k(.9876)-C_1>.0000147$ for the population and
$k(.9876)-C_1>.0000048$ for the finite case. Subtracting the retained
$kQ_\beta(r,e)$ reserve proves (SFA84.1).
:::

(sec-sfa84-scope)=
## 4. Remaining signed derivative and precise accepted scope

:::{prf:proposition} The unabsorbed complete second response is retained exactly
:label: prop-sfa84-second

Record80's complete physical differential satisfies the exact equality

$$
\begin{split}
Q_\beta(R,D[Z_b+aF_2])-Q_\beta(R,DZ_b)
={}&2a\langle DZ_b+\beta R,DF_2\rangle+a^2|DF_2|^2,\\
F_2={}&B_2-L_2W.
\end{split}
\tag{SFA84.4}
$$

The frozen result retains its actual expectation as an unclosed signed
quantity. It does not claim that (SFA84.1) contracts the physical
kinetic update or a normalized alive readout.
:::

:::{prf:proof}

Expand the velocity quadratic and its cross term at each actual outcome.
Every term in (SFA84.4) follows without conditional independence. The
full first and conditional Gaussian second-force estimates in the
accepted dependencies give their stated integrability. The actual second
count alignment, actual spatial graph derivative and the native cap
remain together inside $DF_2$ and its signed cross term.

An endpoint transport coupling cannot be inferred by integrating the
auxiliary pair: no separate transition map with that pair as its
differential was asserted or constructed. Actual preparation, original
source/Haar/revival changes and each own alive or swarm-survival
normalization still need their own complete comparison. A finite budget
event requires its actual mixed exceptional displacement charge; future
survival does not leave fresh Gaussian noises. None of these charges is
hidden in the two accepted numerical gaps.
:::

:::{prf:remark} The norm-relaxation obstruction has only its algebraic scope
:label: rem-sfa84-relaxation

The separate relaxation in Section 5 of record80 is correctly scoped.
Its selected $D=.1I$ satisfies $D^2\le.795I$ and $D\ge.0989I$.
The permitted positive absolute residuals give
$\det(I-T)<.000135-(.0392)(.01394)<0$. Thus that real relaxed
matrix has an eigenvalue greater than one and cannot contract in any
fixed positive definite quadratic norm, even after repeated application.

These residual directions have not been shown to occur under the actual
own Gaussian providers. The actual first-force/count/cap compatibility
retained by the signed consumer supplies additional information absent
from this relaxation. The algebraic example consequently does not refute
the actual physical or alive-law target; it excludes only a proof based
on those standalone absolute residual constraints.
:::

(sec-sfa84-verification)=
## 5. Audit verification record

The complete embedded rational certificate in the frozen source was
executed successfully. An independent symbolic-rational construction
also verified all three multiplier minors and all Bernstein coefficient
floors across sixteen intervals for both parameter pairs. It independently
checked the four derivative norm bounds at the interval endpoints using
their convex squared-norm polynomials and the corrected perturbation
reserve. No float eigenvalue or Monte Carlo approximation was used to
establish an inequality. The small displayed numerical minima from the
audit are diagnostics only; the accepted comparisons are exact rationals.

The audit touched only this new record. Research79, all previously
accepted dependencies, algorithm code and main chapter proofs are
preserved. The complete signed second response remains the next physical
absorption obligation, followed by actual preparation and separately
normalized alive laws.
