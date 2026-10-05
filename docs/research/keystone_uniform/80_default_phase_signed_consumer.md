# Signed first-provider absorption under the actual noisy cap

(sec-sfc80-register)=
## 1. Actual default differential and the isolated second response

:::{prf:definition} Actual cap with a first-provider comparison differential
:label: def-sfc80-register

Retain the unchanged harmonic count update of research68,74,76,77:
$d=3$, $t=.02$, $a=.006$, $c=e^{-.04}$, $V=2$,
$\beta=.04$, and

$$
b=t(1+c),\quad a_x=1-tb,\quad z_v=c-tb,\quad
r_H=(1-t^2)b,\qquad Q_\beta(r,p)=|r|^2+|p|^2+2\beta r\cdot p.
\tag{SFC80.1}
$$

Use any square-integrable prepared comparison differential $(r,p)$.
For a population prepared law assume $|P|\le4$, its actual velocity RMS
is at most $.55$, and its actual own second-stage velocity provider has
first moment at most $.70$. The full recipient-jitter source plans of
research73,76 satisfy these hypotheses after the declared velocity burn.
No individual position bound, narrow velocity band, or independence of
the prepared displacement and velocity is imposed.

For a fixed finite prepared array assume the actual pre-OU budgets of
research74: $|P_i|\le4$, RMS at most $.56$, and normalized positional
second moment at most $12.25$. Its actual count denominator is $N$.
For $N=1$ the count field is zero and the cap statement applies directly.
Random finite preparations can be treated conditionally on their complete
pre-OU array on this budget event; the exceptional displacement moments
remain outside the result.

Let $L_1,L_2$ be the two actual own count operators, with their actual
first and correlated second-stage conductances. Write

$$
e=(I-aL_1)p,\qquad f=B_1=-L_{\dot k_1}P,\qquad E=e+af,
$$

$$
R=a_xr+bE,\qquad W=c(E-tr),\qquad Z_b=z_vE-r_Hr,
\qquad F_2=B_2-L_2W.
\tag{SFC80.2}
$$

The actual cap derivative is $D=DC_V(z)$ at the unchanged complete
own noisy precap velocity. Thus $D$ includes both own providers,
the uncapped correlated OU stage, the actual second graph and all
Gaussian tails. The complete physical differential is

$$
(\dot x^+,\dot v^+)=(R,D[Z_b+aF_2]).
\tag{SFC80.3}
$$

The pair $(R,DZ_b)$ below is an auxiliary differential inside this
actual account. It is not asserted to be the differential of a
separate algorithm or an endpoint transport map. The entire second
response $aDF_2$ is isolated explicitly rather than removed from
the physical transition. All quantities $r,p,e,f,R,W,Z_b$ are fixed
before fresh OU conditional on the complete preparation. Only $D,F_2$
still depend on that fresh noise.
:::

(sec-sfc80-certificate)=
## 2. A rational cap-deficit multiplier

:::{prf:lemma} Signed scalar spectral certificate
:label: lem-sfc80-spectral

Define the matrices and rows

$$
G=\begin{pmatrix}1&\beta&0\\\beta&1&0\\0&0&0\end{pmatrix},
\quad Q_K=\operatorname{diag}(K^2,0,-1),
$$

$$
\mathsf R=(a_x,b,ba),\qquad
\mathsf Z=(-r_H,z_v,z_va),\qquad \tau=.00015478,
$$

$$
H=\begin{pmatrix}
.000274&-.001361&-.0001797\\
-.001361&.922237&.0012396\\
-.0001797&.0012396&.0001181
\end{pmatrix}.
\tag{SFC80.4}
$$

Every displayed decimal is an exact terminating rational. The matrix
$H$ is positive definite. For either of the two pairs

$$
(K,k)=(2.76185,.00002)\quad\hbox{or}\quad
(K,k)=(2.76792,.00001),
\tag{SFC80.5}
$$

the following matrix is positive semidefinite for every $d\in[0,1]$:

$$
\begin{split}
\mathcal M_{K,k}(d)={}&(1-k)G-\mathsf R^{\mathsf T}\mathsf R
-d^2\mathsf Z^{\mathsf T}\mathsf Z
-\beta d(\mathsf R^{\mathsf T}\mathsf Z+
                       \mathsf Z^{\mathsf T}\mathsf R)\\
&-\tau Q_K+(d^2-159/200)H.
\end{split}
\tag{SFC80.6}
$$

This is a certificate for the signed cross term. It does not replace
the cap by an independent scalar derivative.
:::

:::{prf:proof}

The leading principal minors of $H$ are exactly

$$
.000274,\qquad .000250840617,\qquad .00000000002862818517,
$$

so Sylvester's criterion gives positive definiteness. For the original
exponential coefficient use the rational interval

$$
c_- =.960789439152<c<c_+=.960789439153.
\tag{SFC80.7}
$$

Indeed the degree-13 and degree-12 alternating partial sums of
$e^{-.04}$ give a lower and an upper bound, respectively, strictly
inside these two endpoints. Their comparison is exact rational
arithmetic. Set $c_*=(c_-+c_+)/2$ and $\eta=10^{-9}$.
At $c_*$, form the leading principal minors of
$\mathcal M_{K,k}(d)-\eta I_3$. They are polynomials of degree
at most $2,4,6$. On each of the sixteen subintervals
$[j/16,(j+1)/16]$, express each polynomial in its Bernstein basis.
The smallest coefficient in all sixteen bases has the following
strict rational lower bounds:

| $(K,k)$ | First minor | Second minor | Third minor |
|---|---:|---:|---:|
| $(2.76185,.00002)$ | $.0001495$ | $.0000360$ | $10^{-10}$ |
| $(2.76792,.00001)$ | $.0001543$ | $.0000373$ | $1.5\,10^{-10}$ |

Section 6 gives the complete finite exact-arithmetic verification.
For a polynomial $p(d)=\sum_{i=0}^n p_i d^i$, its power coefficients
after the substitution $d=l+(u-l)v$ are

$$
q_i=\sum_{j=i}^n p_j {j\choose i}l^{j-i}(u-l)^i.
$$

Its degree-$n$ Bernstein coefficients are
$b_k=\sum_{i=0}^k q_i {k\choose i}/{n\choose i}$.
The Bernstein basis functions are nonnegative and sum to one on
$[0,1]$, so the displayed strict bounds prove positivity of all
three minors throughout $[0,1]$. Sylvester's criterion proves
$\mathcal M_{K,k}(d;c_*)\ge\eta I_3$.

To retain the exact original $c$, bound the derivative of (SFC80.6)
with respect to $c$. Throughout $.96\le c\le.961$,

$$
|\mathsf R|<2,\quad |\mathsf Z|<2,\quad
|\partial_c\mathsf R|<.03,\quad |\partial_c\mathsf Z|<1.1.
$$

Differentiating its rank-one terms and using $0\le d\le1$ gives

$$
\|\partial_c\mathcal M_{K,k}\|_{\rm op}
\le 2(2)(.03)+2(2)(1.1)
 +2(.04)[(.03)(2)+(2)(1.1)]<5.
$$

The remaining matrices are independent of $c$. Thus replacing $c_*$
by the actual $c$ changes the matrix by operator norm less than
$5\cdot10^{-12}<\eta$. This proves (SFC80.6) for the original
coefficient, with no floating-point assumption about its spectrum.
:::

(sec-sfc80-first)=
## 3. Absorption of the complete first spatial response

:::{prf:theorem} Signed first-provider and actual cap consumer
:label: thm-sfc80-first-consumer

Under {prf:ref}`def-sfc80-register`, the actual auxiliary differential
satisfies, for the population,

$$
\mathbb E Q_\beta(R,DZ_b)
\le (1-.0000147)\mathbb E Q_\beta(r,p).
\tag{SFC80.8}
$$

For every fixed finite prepared array with its declared actual budgets,
the normalized full-OU expectation satisfies

$$
\mathbb E_\xi\langle Q_\beta(R,DZ_b)\rangle_N
\le (1-.0000048)\langle Q_\beta(r,p)\rangle_N.
\tag{SFC80.9}
$$

The rates do not depend on $N$ and contain no particle residual. These
are signed differential consumers, with their own exact scope in
(SFC80.2)--(SFC80.3). Neither is the complete kinetic or alive-law
contraction statement.
:::

:::{prf:proof}

**Step 1: apply the spectral certificate before averaging.**
Set the three-vector of physical vectors to $q=(r,e,f)$. The cap
derivative $D$ is self-adjoint with spectrum in $[0,1]$. For each
realized root diagonalize this actual $D$ and apply (SFC80.6) to
the three scalar coordinates in each of its eigendirections.
Summing gives the pointwise inequality

$$
\begin{split}
&(1-k)Q_\beta(r,e)-Q_\beta(R,DZ_b)\\
&\quad\ge\tau(K^2|r|^2-|f|^2)
-\sum_{i,j=1}^3 H_{ij}
       \langle q_i,(D^2-159/200\,I)q_j\rangle.
\end{split}
\tag{SFC80.10}
$$

This calculation allows arbitrary orientation and correlations of the
three prepared vectors and of $D$. It never assigns $D$ a fixed
scalar value or commutes it with either count graph.

**Step 2: consume the two exact quadratic constraints.**
Since $H\ge0$, choose its square root and write
$T_l=\sum_i(H^{1/2})_{li}q_i$. Each $T_l$ is fixed before OU.
The conditional cap defect of research74 therefore gives

$$
\mathbb E_\xi\sum_l\langle T_l,
       (D^2-159/200\,I)T_l\rangle\le0
\tag{SFC80.11}
$$

at each complete preparation. This is the full signed matrix term
in (SFC80.10), not a product of averaged factors. Research77 gives
$\|f\|_2\le K\|r\|_2$ with the relevant constant in
(SFC80.5), including actual normalized finite denominators.
Averaging (SFC80.10) consequently proves

$$
\mathbb E Q_\beta(R,DZ_b)
\le(1-k)\mathbb E Q_\beta(r,e).
\tag{SFC80.12}
$$

For a population, its outer averaging includes the entire original
recipient-jitter law and accepted source/component/Haar plan. For a
fixed finite array its force bound is the exact normalized array
bound conditional on that preparation. The original full independent
OU array is averaged without a survival or empirical-OU restriction.

**Step 3: retain the actual first-count response.**
The actual symmetric count operator obeys $0\le L_1\le I$.
Since $e=p-aL_1p$,

$$
\begin{split}
\mathbb E[Q_\beta(r,e)-Q_\beta(r,p)]
\le{}&-a(2-a)\|L_1^{1/2}p\|_2^2
       -2\beta a\langle L_1^{1/2}r,L_1^{1/2}p\rangle\\
\le{}&\frac{\beta^2a}{2-a}\|r\|_2^2.
\end{split}
\tag{SFC80.13}
$$

The first line uses $L_1^2\le L_1$; completing its square and
$\|L_1^{1/2}r\|_2\le\|r\|_2$ proves the second line.
There is no favorable-sign assumption on the position--velocity
cross term. Since $Q_\beta\ge(1-\beta)(|r|^2+|p|^2)$,

$$
\mathbb E[Q_\beta(r,e)-Q_\beta(r,p)]
\le C_1\mathbb E Q_\beta(r,p),\qquad
C_1=\frac{\beta^2a}{(2-a)(1-\beta)}.
\tag{SFC80.14}
$$

Also $(r,e)-(r,p)=(0,-aL_1p)$ has $Q_\beta$-norm at most
$a\|p\|_2$. The reverse triangle inequality in this genuine
Hilbert norm gives

$$
\mathbb E Q_\beta(r,e)
\ge(1-a/\sqrt{1-\beta})^2\mathbb E Q_\beta(r,p)
>.9876\,\mathbb E Q_\beta(r,p),
\tag{SFC80.15}
$$

with a non-strict final comparison when the input differential is zero.
The exact checks are $a/\sqrt{.96}<.006124$ and
$(1-.006124)^2>.9876$.

Subtract $k\mathbb E Q_\beta(r,e)$ in (SFC80.12), apply
(SFC80.14)--(SFC80.15), and use the rational comparisons

$$
.00002(.9876)-C_1>.0000147,\qquad
.00001(.9876)-C_1>.0000048.
\tag{SFC80.16}
$$

This proves (SFC80.8)--(SFC80.9). Every zero-displacement case
is covered without division or a strict product comparison.
:::

(sec-sfc80-second)=
## 4. The actual complete second-field interface

:::{prf:corollary} Remaining signed second-provider response
:label: cor-sfc80-second-interface

For the actual complete differential (SFC80.3), define

$$
\mathcal J_{2,C}=
2a\mathbb E\langle DZ_b+\beta R,DF_2\rangle
       +a^2\mathbb E|DF_2|^2.
\tag{SFC80.17}
$$

Then the population obeys

$$
\mathbb E Q_\beta(\dot x^+,\dot v^+)
-\mathbb E Q_\beta(r,p)
\le-.0000147\,\mathbb E Q_\beta(r,p)+\mathcal J_{2,C}.
\tag{SFC80.18}
$$

The exact fixed finite-array version has coefficient $.0000048$,
normalized row sums, and expectation over the actual complete
original OU array in (SFC80.17). The quantity $F_2$ contains both
the second spatial provider $B_2$ and the actual second alignment
$-L_2W$, under their same noisy graph and cap.
:::

:::{prf:proof}

Expansion at each actual outcome gives

$$
Q_\beta(R,D[Z_b+aF_2])-Q_\beta(R,DZ_b)
=2a\langle DZ_b+\beta R,DF_2\rangle+a^2|DF_2|^2.
$$

Average this exact equality and apply the theorem. Bounded prepared
velocities, the first-force estimate and the conditional full-Gaussian
second-force estimate in research68 give the required $L^2$
integrability. No cap expectation is applied to the OU-dependent
vector $F_2$, and no second provider is replaced by its mean.
:::

:::{prf:remark} Scope of the completed signed absorption
:label: rem-sfc80-scope

The first spatial response, its force square, the first count response,
and their complete signed cross term against the actual noisy cap
are absorbed in a positive differential margin. The certificate uses
the full centered first-force bound and the conditional cap-square
deficit. It works for unrestricted positional shapes under the stated
prepared budgets, rather than a pointwise velocity-band class.

The entire remaining expression (SFC80.17) is still present. In
particular the cap-force cross term has not been replaced by its
absolute consumer or assigned a favorable sign. A full kinetic
margin would require a proved upper bound for this signed expression
that fits (SFC80.18), or a stronger account retaining more of the
positive reserves in (SFC80.10)--(SFC80.13).

The auxiliary differential has not been integrated into a transport
claim. Actual accepted preparation/source/Haar/revival comparison,
terminal marking, each own survival and alive denominator, and class
invariance or a delayed block remain separate obligations. For random
finite preparations the actual mixed exceptional displacement charge
and each weighted survival deficit remain necessary. These statements
do not identify a QSD or prove a default nonlinear alive-law mixing
rate. The harmonic force calculation is not a Rastrigin calculation.
:::

(sec-sfc80-norm-obstruction)=
## 5. A precise limitation of the absolute residual interface

:::{prf:proposition} The residual-norm relaxation admits an unstable matrix
:label: prop-sfc80-norm-relaxation

The standalone scalar cap sector $0\le D\le I$,
$D^2\le159/200$, the scalar mean lower bound $D\ge.0989$, and
the two absolute residual estimates of research76 do not suffice
to establish contraction in any fixed quadratic phase norm.
This is a statement about that algebraic relaxation, not about the
actual cap/provider map or delayed alive-law mixing.
:::

:::{prf:proof}

Choose in the relaxation $D=.1I$, $A_1=A_2=I$ and collinear
displacements. Use the permitted residuals
$\delta R=.000650r$ and
$\delta C=.01787r+.000344p$ with their positive orientations.
The resulting scalar linear comparison matrix is

$$
T=\begin{pmatrix}
a_x+.000650&b\\
-.1r_H+.01787&.1z_v+.000344
\end{pmatrix}.
\tag{SFC80.19}
$$

All three scalar cap inequalities hold for $D=.1$. Write
$\alpha=T_{11}$, $\delta=T_{22}$, $u=T_{12}$ and
$v=T_{21}$. The retained coefficient intervals give

$$
0<1-\alpha<.000135,\quad 0<1-\delta<1,\quad
u>.0392,\quad v>.01394.
$$

Consequently

$$
\det(I-T)=(1-\alpha)(1-\delta)-uv
<.000135-(.0392)(.01394)<0.
\tag{SFC80.20}
$$

The characteristic polynomial of the real $2\times2$ matrix is
negative at one and positive for sufficiently large real arguments,
so it has a real eigenvalue $\lambda>1$. For its real eigenvector
$h\ne0$ and any positive definite phase matrix $G_*$,
$h^{\mathsf T}T^{\mathsf T}G_*Th=\lambda^2h^{\mathsf T}G_*h$.
Thus no fixed positive definite quadratic norm contracts every map
allowed by this relaxation, nor can iterating this relaxed matrix
produce a delayed contraction.

The selected residuals have only the allowed norms; they have not
been asserted to arise from the actual Gaussian count fields. The
real $D$ also has its own full Gaussian distribution. This example
therefore identifies the information lost by using the absolute
interface alone. It is not a lower bound on actual feedback or a
counterexample to any actual alive-law estimate.
:::

(sec-sfc80-exact-checks)=
## 6. Complete finite rational verification

:::{prf:remark} Exact coefficient certificate and retained dependencies
:label: rem-sfc80-exact-checks

The following finite certificate verifies all numerical signs in the
spectral lemma, including all Bernstein coefficients. Polynomial
lists store power coefficients. The checks use rational arithmetic
only; floating-point eigenvalues are not a proof input.

```python
from fractions import Fraction as F
from math import comb, factorial


def add(x, y):
    n = max(len(x), len(y))
    return [
        (x[i] if i < len(x) else F(0))
        + (y[i] if i < len(y) else F(0))
        for i in range(n)
    ]


def subtract(x, y):
    return add(x, [-v for v in y])


def multiply(x, y):
    result = [F(0)] * (len(x) + len(y) - 1)
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            result[i + j] += a * b
    return result


def bernstein(poly, left, right):
    n = len(poly) - 1
    powers = [
        sum(
            (
                poly[j] * comb(j, i) * left ** (j - i)
                * (right - left) ** i
                for j in range(i, n + 1)
            ),
            F(0),
        )
        for i in range(n + 1)
    ]
    return [
        sum(
            (powers[i] * F(comb(k, i), comb(n, i)) for i in range(k + 1)),
            F(0),
        )
        for k in range(n + 1)
    ]


def determinant(matrix):
    first = multiply(
        matrix[0][0],
        subtract(multiply(matrix[1][1], matrix[2][2]),
                 multiply(matrix[1][2], matrix[2][1])),
    )
    second = multiply(
        matrix[0][1],
        subtract(multiply(matrix[1][0], matrix[2][2]),
                 multiply(matrix[1][2], matrix[2][0])),
    )
    third = multiply(
        matrix[0][2],
        subtract(multiply(matrix[1][0], matrix[2][1]),
                 multiply(matrix[1][1], matrix[2][0])),
    )
    return add(subtract(first, second), third)


t, a, beta = F('.02'), F('.006'), F('.04')
lower, upper = F('.960789439152'), F('.960789439153')
partial_13 = sum(
    (F((-1) ** j) * F('.04') ** j / factorial(j) for j in range(14)),
    F(0),
)
partial_12 = partial_13 + F('.04') ** 13 / factorial(13)
assert lower < partial_13 < partial_12 < upper
c = (lower + upper) / 2
b, ax = t * (1 + c), 1 - t * t * (1 + c)
zv, rh = c - t * b, (1 - t * t) * b
tau, eta = F('.00015478'), F('1e-9')
H = [
    [F('.000274'), F('-.001361'), F('-.0001797')],
    [F('-.001361'), F('.922237'), F('.0012396')],
    [F('-.0001797'), F('.0012396'), F('.0001181')],
]
assert H[0][0] > 0
assert H[0][0] * H[1][1] - H[0][1] ** 2 > 0
assert determinant([[[v] for v in row] for row in H])[0] > 0
G = [[F(1), beta, F(0)], [beta, F(1), F(0)], [F(0)] * 3]
R, Z = [ax, b, b * a], [-rh, zv, zv * a]
cases = [
    (F('2.76185'), F('.00002'),
     [F('.0001495'), F('.0000360'), F('1e-10')], F('.0000147')),
    (F('2.76792'), F('.00001'),
     [F('.0001543'), F('.0000373'), F('1.5e-10')], F('.0000048')),
]
for K, k, floors, final_gap in cases:
    Q = [[K * K, F(0), F(0)], [F(0)] * 3, [F(0), F(0), F(-1)]]
    M = [
        [
            [
                (1 - k) * G[i][j] - R[i] * R[j] - tau * Q[i][j]
                - F('.795') * H[i][j] - (eta if i == j else 0),
                -beta * (R[i] * Z[j] + Z[i] * R[j]),
                H[i][j] - Z[i] * Z[j],
            ]
            for j in range(3)
        ]
        for i in range(3)
    ]
    minors = [
        M[0][0],
        subtract(multiply(M[0][0], M[1][1]),
                 multiply(M[0][1], M[1][0])),
        determinant(M),
    ]
    for poly, floor in zip(minors, floors):
        for j in range(16):
            assert min(bernstein(poly, F(j, 16), F(j + 1, 16))) > floor
    cost = beta ** 2 * a / ((2 - a) * (1 - beta))
    assert k * F('.9876') - cost > final_gap
assert a ** 2 / (1 - beta) < F('.006124') ** 2
assert (1 - F('.006124')) ** 2 > F('.9876')
assert F('.000135') - F('.0392') * F('.01394') < 0
print('All signed first-provider/cap rational comparisons pass.')
```

The accepted dependencies remain unchanged:

| Record | SHA-256 |
|---|---|
| 67 | `1541e58b1f499e2528e9358651c2f701af651761f6fa9b05a3968ec43c3fd821` |
| 68 | `ac2799245012b378edb22326fb4b9035b6dfa9aab47de5dba8fe66022e40af58` |
| 69 | `012f6f1f2671ddac98e10b4d63abca7662a5759deeb8267c475a399425a01a10` |
| 73 | `3d8338a7cd399f9d746ae6b9bac8e8f847fdb2ba33cec83be235ba46e022a67a` |
| 74 | `fe58c2ccd64571ac2979bd3d36877f2aef0c6528e5d234cd13edd77cca0b5225` |
| 76 | `d06b3e1a79a491e0308a976baba5f13b1a68dda90a4e8902b056523d41d68060` |
| 77 | `831289ee74b8679d274856bc3f3e89c9837b4ecbd90ff95a68dcc982a412faf4` |
:::
