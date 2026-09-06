# Direct Observables and Standard Model Representations on the Fractal Set

(sec-sm-introduction)=
## 1. Recorded Data and the Direct Observable Formulation

:::{div} feynman-prose
Start with the quantities the gas records: forces, velocities, fitness, and
companions. The Fractal Set reconstruction supplies these inputs; explicit
formulas turn them into complex vectors and their contractions. The complete
Gram and determinant coordinates recover the descriptor configuration up to
a common special-unitary frame change.

The complete record also lets us follow the update itself. Decode the state,
run the implemented step with its random inputs, and encode the result.
{prf:ref}`thm-sm-instantiated-record-transition` makes this recipe an exact
transition kernel. Every integrable finite history observable covered by the
record keeps its expectation, including the masks used to select samples.

The derivative and LSI bounds then control the specified fields and laws.
On its identified LSI law, the equilibrium energy construction supplies a
further evolution with its own generator and time coordinate. The exterior
construction carries record observables and their evolution to antisymmetric
replicas of the whole swarm. Comparing these constructions with a proposed
coupled field action uses the measure and generator calculations below.
Dirac matrices provide an optional representation of the operator algebra.
:::

:::{prf:definition} Scope, dimensions, and phase conventions
:label: def-sm-scope

Let $\mathcal F$ be the recorded Fractal Set of {doc}`01_fractal_set`.
CST edges record episode succession; clone ancestry is additional information.
The primary construction in this chapter is the direct observable map
of {prf:ref}`def-sm-direct-observable-law`: complex color vectors, companion
amplitudes, pair contractions, determinants, and triangle products are
computed from recorded data. Their finite law is the pushforward of the
specified record law. This construction uses commuting numerical fields.
The exterior-algebra and Dirac constructions in
{ref}`sec-sm-matter-sector` are additional field representations.
Use $N$ for population size, $d$ for latent spatial dimension, $n$ for a
chosen color-fiber rank, and $n_g$ for a chosen generation multiplicity.
These integers describe different spaces. The Standard Model representation
below uses $n=3$ and may be repeated $n_g$ times.

Write $F_i$ or $V_i$ for recorded fitness. An assigned phase must have a
dimensionless exponent: $\hbar_{\mathrm{eff}}$ has the units of an assigned
edge action, while a dimensionless score uses a dimensionless phase scale
$h_S>0$. Equating these scales requires an explicit nondimensionalization.
A node potential difference and an accumulated path action are distinct
quantities unless their equality has been established.

Comparison links have the convention of {prf:ref}`def-lqft-link-convention`:
$U_{ij}:V_j\to V_i$, $\psi_i\mapsto\Omega_i\psi_i$, and
$U_{ij}\mapsto\Omega_iU_{ij}\Omega_j^{-1}$.
A continuum geometry is supplied under
{prf:ref}`assm-cst-continuum-geometry`. Its sampling and reconstruction
requirements are those of {doc}`03_lattice_qft` and
{prf:ref}`cor-continuum-consistency-conditional`.
:::

:::{prf:remark} Relation to the agent volume
:label: remark-sm-vol1-connection

The agent volume also organizes phase, doublet, and component representations.
Its chart-fission calculation supplies a quartic normal-form calculation
reused in {prf:ref}`thm-sm-higgs-isomorphism`. Applying that normal form to
Fractal Gas requires a reduction of its own dynamics with the stated
coefficients. A common representation or polynomial potential identifies
that mathematical structure; it does not identify the complete dynamics
or establish a unique physical theory.
:::

(sec-sm-gauge-principle)=
## 2. Internal Representations and Exact Covariance

:::{div} feynman-prose
Imagine writing the same collection of vectors in another orthonormal basis.
Their components move, while their inner products stay fixed. Preserve their
complex oriented volumes as well, and the allowed transformations are exactly
special unitary. For two components, the oriented volume is the alternating
doublet contraction; for three, it is a three-vector determinant.

The companion draw makes these observables fluctuate. Fix the current state
and fitness inputs, and compare the amplitudes produced by the eligible
companions. Different distances give different amplitude magnitudes. When
both companions have positive selection probability, this difference gives
a strictly positive variance of the recorded component and hence of its
doublet, as calculated below.

Transport follows the Fractal Set's CST, IG, and IA edge operations. An
interaction triangle compares its IA and IG matrices, with the CST factor
retained outside temporal gauge. Its Wilson defect is calculated directly
from those transports in {prf:ref}`prop-sm-attribution-holonomy-defect`.
The complete companion, cloning, and kinetic update supplies the history
law used for the doublets and the recorded transport observables.
:::

### Companion amplitudes and phases

:::{prf:theorem} Phase freedom of companion amplitudes
:label: thm-sm-u1-emergence

For nonnegative weights $w_{ik}$ with positive row sum define

$$
P_i(k)=\frac{w_{ik}}{\sum_lw_{il}},\qquad
\psi_i(k)=\sqrt{P_i(k)}e^{i\theta_{ik}}.
$$

Then $\sum_k|\psi_i(k)|^2=1$, and independent changes of the phases leave
the companion probabilities unchanged. A chosen common rephasing at each
vertex, $\psi_i\mapsto e^{i\alpha_i}\psi_i$, is a $U(1)$ action. Links
$U_{ij}\in U(1)$ transforming as
$U_{ij}\mapsto e^{i\alpha_i}U_{ij}e^{-i\alpha_j}$ give covariant
comparisons $U_{ij}\psi_j-\psi_i$.

The diversity kernel may be used for the weights:
$w_{ik}=\exp[-d_{\mathrm{alg}}(i,k)^2/(2\epsilon_d^2)]$, with its actual
allowed-companion set and normalization. The recorded IG, CST, and IA phase transports are those of
{prf:ref}`def-fractal-set-gauge-connection`; their ordered interaction
loop is evaluated in {prf:ref}`def-fractal-set-wilson-loop`.

For $V_i+\varepsilon_{\mathrm{clone}}>0$, a common fitness shift $b$ with
$V_i+b+\varepsilon_{\mathrm{clone}}>0$ preserves fitness differences and
score signs, but changes scores according to

$$
S_i^{(b)}(j)
 =\frac{V_i+\varepsilon_{\mathrm{clone}}}
        {V_i+b+\varepsilon_{\mathrm{clone}}}S_i(j).
$$

Thus fixed-source score order is preserved. Acceptance probabilities based
on score magnitude need not be preserved. A phase representation of
companion probabilities does not imply local rephasing symmetry of the full
particle update.
:::

:::{prf:proof}
Normalization follows by summing $P_i(k)$. Taking moduli removes all phases.
Substitution of the link transformation leaves the comparison multiplied by
$e^{i\alpha_i}$. Substitution of $V_i+b$ into the score gives the displayed positive row
factor. For example a score $1/2$ becomes $1/4$ if that factor is $1/2$;
its Bernoulli acceptance probability changes under the usual unclipped
score rule. $\square$
:::

:::{prf:theorem} A normalized doublet and SU(2) frame changes
:label: thm-sm-su2-emergence

For recorded cloning-companion probabilities $p_{ij},p_{ji}$ with positive
sum define

$$
z_{ij}=\frac1{\sqrt{p_{ij}+p_{ji}}}
\begin{pmatrix}
\sqrt{p_{ij}}e^{iS_i(j)/h_S}\\
\sqrt{p_{ji}}e^{iS_j(i)/h_S}
\end{pmatrix}.
$$

This is a unit vector in $\mathbb C^2$. On a chosen doublet fiber,
complex-linear maps preserving the Hermitian product and a fixed determinant
volume form constitute $SU(2)$. Its Hermitian generators
$T^a=\sigma^a/2$ obey
$[T^a,T^b]=i\epsilon^{abc}T^c$ and
$\operatorname{Tr}(T^aT^b)=\delta^{ab}/2$.
Comparison links in this group give the covariance stated above.

The basis and determinant structure are part of the field representation.
A general unitary mixing of a doublet changes its individual component
probabilities; invariance of those probabilities alone does not select
$SU(2)$. A cloning companion choice is also distinct from a successful
cloning event.
:::

:::{prf:proof}
The squared norm is $(p_{ij}+p_{ji})/(p_{ij}+p_{ji})=1$.
Preservation of the Hermitian form is $U^\dagger U=I$; preservation of the
chosen volume form is $\det U=1$. Their intersection is the definition of
$SU(2)$. Multiplication of the Pauli matrices gives
$\sigma^a\sigma^b=\delta^{ab}I+i\epsilon^{abc}\sigma^c$, proving the
commutator and trace identities. Gauge covariance follows by cancellation
of the basis change at the transported endpoint.
For the probability assertion, an $SU(2)$ rotation sends $(1,0)$ to
$(\cos\vartheta,\sin\vartheta)$, changing its component probabilities.
$\square$
:::

### Viscous covariance and internal color

:::{prf:theorem} Orthogonal covariance and a chosen SU(n) representation
:label: thm-sm-su3-emergence

Consider the recorded viscous force

$$
F_i^{\mathrm{visc}}=\nu\sum_j K_{ij}(v_j-v_i).
$$

If the scalar kernel $K_{ij}$ is unchanged under a simultaneous orthogonal
change $(x_i,v_i)\mapsto(Ox_i,Ov_i)$, then
$F_i^{\mathrm{visc}}\mapsto OF_i^{\mathrm{visc}}$.
For $F_i^{\mathrm{visc}}\ne0$, the componentwise encoding

$$
c_i^{(a)}=\frac{F_i^{\mathrm{visc},a}}{\|F_i^{\mathrm{visc}}\|}
 \exp\!\left(\frac{imv_i^a\ell_0}{\hbar_{\mathrm{eff}}}\right)
$$

has unit norm in $\mathbb C^d$. The encoding is generally nonlinear under
orthogonal mixing of components. It therefore requires an additional
representation map before it can be used as a covariant color field.
At zero force it is undefined; any zero-force replacement must be specified.
Dividing by $\sqrt{\|F_i\|^2+\delta^2}$ instead gives norm at most one,
not a unit vector.

Independently, chosen Hermitian fibers $\mathbb C^n$ with a determinant
volume form admit $SU(n)$ frame changes. Setting $n=3$ supplies the color
representation used below. The group identity is

$$
U(n)\cong[U(1)\times SU(n)]/\mathbb Z_n;
$$

quotienting $U(n)$ by all scalar phases gives $PU(n)$, not $SU(n)$.
The determinant-one choice and an identification with the fitness phase
are separate data.
:::

:::{prf:proof}
The force covariance follows by moving $O$ through its linear sum.
The phase factors have modulus one, so normalization gives $\|c_i\|=1$.
To test covariance of the encoding, take $d=2$, $F=(1,0)$ and $v=(p,0)$.
Rotate by $45$ degrees. Encoding after rotation gives
$2^{-1/2}(e^{iap/\sqrt2},e^{iap/\sqrt2})$, where
$a=m\ell_0/\hbar_{\mathrm{eff}}$, whereas rotating the encoded vector
gives $2^{-1/2}(e^{iap},e^{iap})$. They differ for generic $ap$.

The group statement follows from the surjective homomorphism
$(z,U)\mapsto zU$. Its kernel consists of
$(z,z^{-1}I)$ with $z^n=1$. Surjectivity follows by choosing an $n$th root
of the determinant of any unitary matrix. The scalar-phase quotient is by
definition $PU(n)$. $\square$
:::

:::{prf:corollary} Product representations and their faithful group
:label: cor-sm-gauge-group

Choose separate phase, weak-doublet, and color factors, with gauge action
on a tensor product given by

$$
(z,B,A)\cdot\psi=z^q(A\otimes B)\psi,
\qquad (z,B,A)\in U(1)\times SU(2)\times SU(n),
$$

where $q\in\mathbb Z$ is a chosen charge in this normalization.
The factor actions commute. Their faithful action is the product divided by
the subgroup acting trivially on every chosen field representation.
For the conventional $n=3$ representation constructed in
{prf:ref}`thm-sm-so10-isomorphism`, this quotient is
$[SU(3)\times SU(2)\times U(1)]/\mathbb Z_6$.

Independent fiber factors yield this product action even when their
coefficients depend on the same recorded fitnesses. Statistical independence
of the three particle mechanisms is neither assumed nor inferred.
:::

:::{prf:proof}
Matrices acting on distinct tensor factors commute. The first isomorphism
theorem identifies the image of a representation with its domain modulo its
kernel. The $\mathbb Z_6$ calculation is given explicitly in
{prf:ref}`thm-sm-so10-isomorphism`. $\square$
:::

(sec-sm-direct-observables)=
## 3. Direct Color and Companion Observables

:::{div} feynman-prose
Keep every pair inner product and every complex determinant. You now have
enough information to reconstruct the normalized vectors up to their common
SU(2) or SU(3) frame, including configurations whose vectors fail to span
the whole space. The proofs below construct that correspondence and carry
the probability law through it. Every integrable correlation expressed in
these invariant coordinates retains its value. The complete encoded state
also has the explicit transition of
{prf:ref}`thm-sm-instantiated-record-transition`. For a selected collection
of channels, the memory calculation keeps the effect of discarded state
coordinates. The subsequent prediction construction extends those channels
using the actual update until their observable space is preserved by it.

A frame average combines many coordinates into one number. Retain the direct
descriptors and auxiliary data before forming these averages. In the
three-color implementation, retain the validity mask too: clamping a small
norm can leave a nonzero numerical vector, whereas the masked theoretical
extension assigns zero to an invalid sample.

The statistical estimate must follow the same readout. For a smooth channel,
we can pull its gradient back to the recorded state and use the established
LSI. A hard mask may introduce a jump, so its estimate instead uses the
applicable bounded-observable or temporal covariance result. Counting walkers
alone does not supply a variance bound for an arbitrary whole-swarm channel.
:::

### Recorded fields and their probability law

:::{prf:definition} Direct observable map and record law
:label: def-sm-direct-observable-law

Fix a finite observation schedule, all reconstruction parameters, and the
complete law $\mathbb P_{\mathrm{rec}}$ of the recorded particle history
$\mathcal F$. This law includes companion choices, cloning indicators,
kinetic noise, and any survival conditioning. For the direct three-component color path in this chapter set $d=3$ and
use $c_i\in\mathbb C^3$ from {prf:ref}`thm-sm-su3-emergence` on valid force
samples. The general routine returns $\mathbb C^d$; a different latent dimension
requires a separately specified map into $\mathbb C^3$ before these determinant
channels are evaluated. The baryon routine enforces three components.

Distinguish the raw numerical color from its masked extension. In real
arithmetic, with the routine's threshold $\delta_c=10^{-12}$ in its numerical
force units, put

$$
\widetilde c_i^a=F_i^{\mathrm{visc},a}e^{i\kappa v_i^a},\qquad
c_i^{\mathrm{raw}}=\frac{\widetilde c_i}{\max(\|\widetilde c_i\|,\delta_c)},
\qquad m_i=\mathbf1_{\{\|\widetilde c_i\|>\delta_c\}},
\qquad c_i=m_ic_i^{\mathrm{raw}},
\quad\kappa=\frac{m\ell_0}{\hbar_{\mathrm{eff}}}.
$$

`compute_color_states_batch` returns $c_i^{\mathrm{raw}}$ and $m_i$, rather
than the zero extension $c_i$. Downstream masks implement the latter convention
for the valid contractions. For example $F=(\delta_c/2,0,0)$, $v=0$ gives
$c^{\mathrm{raw}}=(1/2,0,0)$ and $m=0$. The unit-vector formulas below concern
valid samples. Floating-point phase evaluation and normalization retain their
numerical errors; the identities specify the real-arithmetic observable.

Let $\mathscr O(\mathcal F)$ collect the specified direct observables and
their masks at all recorded times. Their finite field law is

$$
\mu_{\mathrm{dir}}=\mathscr O_*\mathbb P_{\mathrm{rec}},\qquad
\langle f\rangle_{\mathrm{dir}}
=\mathbb E_{\mathbb P_{\mathrm{rec}}}f(\mathscr O(\mathcal F))
$$

for bounded measurable $f$, and for integrable $f$ when its moment exists.
At a frame, selected nonnegative weights $w_I$ and validity indicators $m_I$
define the recorded average

$$
\mathcal A_t(O)=
\begin{cases}
\displaystyle\frac{\sum_Iw_Im_IO_I}{\sum_Iw_Im_I},
 &\sum_Iw_Im_I>0,\\
0,&\sum_Iw_Im_I=0.
\end{cases}
$$

The zero-denominator indicator is retained with the average. Different pair
selection, score orientation, weighting, or invalid-sample conventions define
different observable maps. The result is a law of numerical fields before any
Dirac or Grassmann representation is introduced. Its normalization follows
from that of $\mathbb P_{\mathrm{rec}}$. An equilibrium application specifies
whether the law is stationary for a conservative process, a survival-conditioned
history, or a history of a Doob-transformed process.
:::

:::{prf:definition} Direct color contractions
:label: def-sm-direct-color-contractions

For valid color vectors in the common recorded component basis define

$$
q_{ij}=c_i^\dagger c_j,\qquad
b_{ijk}=\det[c_i,c_j,c_k],\qquad
\Pi_{ijk}=q_{ij}q_{jk}q_{ki}.
$$

The standard pair channels are $\operatorname{Re}q_{ij}$ and
$\operatorname{Im}q_{ij}$. The displacement-weighted channels are
$\operatorname{Re}q_{ij}\,r_{ij}$ and
$\operatorname{Im}q_{ij}\,r_{ij}$, with the chosen oriented displacement
$r_{ij}$ or its unit normalization. Determinant channels use a specified
real or imaginary part of $b_{ijk}$. Triangle channels include
$\operatorname{Re}\Pi_{ijk}$ and $1-\operatorname{Re}\Pi_{ijk}$.
On nonzero triangle products, phase channels can also use
$1-\cos(\arg\Pi_{ijk})$ or $\sin^2(\arg\Pi_{ijk})$; zero products require
the implementation's declared phase convention or a mask.

These formulas are the standard modes of
`src/fragile/physics/operators/meson_operators.py`,
`vector_operators.py`, `baryon_operators.py`, and `glueball_operators.py`.
Score-directed and score-weighted modes additionally transform their
orientation and weights according to the configured rule. The color input
is formed in `src/fragile/physics/qft_utils/color_states.py`: the selected
`v_before_clone` frame is paired with its preceding `force_viscous` entry.
That time alignment is part of the observable map.

The names scalar, pseudoscalar, baryon, and glueball label measurement
channels. A spin, charge-conjugation, or physical-particle assignment requires
the corresponding transformations and spectral identification of this law.
:::

:::{prf:theorem} Color invariants and their exact symmetry group
:label: thm-sm-direct-color-invariants

Under a common complex-linear change $c_i\mapsto Ac_i$, every pair
contraction is preserved for all inputs precisely when $A\in U(3)$.
Preserving also the complex determinant $b_{ijk}$ for all triples restricts
the group precisely to $SU(3)$. Consequently $q_{ij}$, $b_{ijk}$, and
$\Pi_{ijk}$ are invariant under common $SU(3)$ frame changes.
For unit vectors,

$$
|q_{ij}|\le1,\qquad |b_{ijk}|\le1,\qquad |\Pi_{ijk}|\le1.
$$

Under independent scalar rephasings $c_i\mapsto e^{i\alpha_i}c_i$,

$$
q_{ij}\mapsto e^{i(\alpha_j-\alpha_i)}q_{ij},\quad
b_{ijk}\mapsto e^{i(\alpha_i+\alpha_j+\alpha_k)}b_{ijk},\quad
\Pi_{ijk}\mapsto\Pi_{ijk}.
$$

Thus $|q_{ij}|^2$, $|b_{ijk}|^2$, and $\Pi_{ijk}$ are invariant under
these independent phases. Real and imaginary pair components generally are
phase dependent. The subgroup preserving a reduced selection of observables
can be larger than the group preserving both full complex contractions.
:::

:::{prf:proof}
**Pair contractions.** For arbitrary $u,v\in\mathbb C^3$,

$$
(Au)^\dagger(Av)=u^\dagger A^\dagger Av.
$$

Preservation gives $u^\dagger(A^\dagger A-I)v=0$. Taking $u=e_a$ and
$v=e_b$ yields $(A^\dagger A-I)_{ab}=0$ for each $a,b$. Thus $A$ is
unitary. Conversely $A^\dagger A=I$ gives equality for every pair.

**Triple contractions.** Put $C=[u,v,w]$. Then

$$
\det[Au,Av,Aw]=\det(AC)=\det(A)\det(C).
$$

Equality at $C=I_3$ forces $\det A=1$. Conversely that equation preserves
every triple determinant. Together with unitarity it is exactly the
definition of $SU(3)$, establishing both inclusions of the symmetry group.

**Bounds.** Cauchy--Schwarz gives
$|c_i^\dagger c_j|\le\|c_i\|\|c_j\|=1$. For the determinant perform
Gram--Schmidt on its three columns. Subtracting earlier-column multiples
does not change the determinant, and the lengths of the resulting
orthogonal columns are at most the original lengths. Their determinant
modulus is their product, at most one. A dependent triple has determinant
zero. The triangle bound follows by multiplying the three pair bounds.

**Independent phases.** With $c'_i=e^{i\alpha_i}c_i$,

$$
q'_{ij}=e^{-i\alpha_i}e^{i\alpha_j}q_{ij},\qquad
b'_{ijk}=e^{i\alpha_i}e^{i\alpha_j}e^{i\alpha_k}b_{ijk},
$$

and therefore

$$
\Pi'_{ijk}
=e^{i[(\alpha_j-\alpha_i)+(\alpha_k-\alpha_j)
                         +(\alpha_i-\alpha_k)]}\Pi_{ijk}
=\Pi_{ijk}.
$$

Taking moduli removes each remaining phase. A common $A=e^{i\alpha}I_3$
preserves all pair contractions and all determinant moduli, while it
multiplies each complex determinant by $e^{3i\alpha}$. This exhibits the
extra transformations retained if determinant phases are discarded.
$\square$
:::

:::{prf:proposition} Projector form and phase of a color triangle
:label: prop-sm-direct-triangle-projectors

Let $P_i=c_ic_i^\dagger$ for unit $c_i$. Then $P_i$ is a rank-one
Hermitian projector and

$$
\Pi_{ijk}=\operatorname{Tr}(P_iP_jP_k).
$$

If all three overlaps are nonzero, their normalized phases
$\ell_{ij}=q_{ij}/|q_{ij}|$ obey $\ell_{ji}=\overline{\ell_{ij}}$ and

$$
\ell_{ij}\ell_{jk}\ell_{ki}=\Pi_{ijk}/|\Pi_{ijk}|.
$$

The phase of this product can be nonzero even though every field is
constructed from vertex data. For example, for
$c_1=(1,0,0)$, $c_2=(1,1,0)/\sqrt2$,
$c_3=(1,i,0)/\sqrt2$, one has $\Pi_{123}=(1+i)/4$.
The ordered projector product is a composite observable. Its factors are
rank-one projectors rather than unitary $SU(3)$ comparison matrices.
:::

:::{prf:proof}
First,

$$
P_i^\dagger=P_i,\qquad
P_i^2=c_i(c_i^\dagger c_i)c_i^\dagger=P_i,\qquad
\operatorname{Tr}P_i=c_i^\dagger c_i=1.
$$

Its image is the one-dimensional span of $c_i$. Multiplying in the given
order gives

$$
P_iP_jP_k
=c_i(c_i^\dagger c_j)(c_j^\dagger c_k)c_k^\dagger
=q_{ij}q_{jk}c_ic_k^\dagger.
$$

Since $\operatorname{Tr}(uv^\dagger)=v^\dagger u$, taking the trace
adds the factor $q_{ki}$, proving the formula. Also
$q_{ji}=\overline{q_{ij}}$, so
$\ell_{ji}=\overline{q_{ij}}/|q_{ij}|=\overline{\ell_{ij}}$.
For a nonzero triangle,

$$
\ell_{ij}\ell_{jk}\ell_{ki}
=\frac{q_{ij}q_{jk}q_{ki}}{|q_{ij}||q_{jk}||q_{ki}|}
=\frac{\Pi_{ijk}}{|\Pi_{ijk}|}.
$$

For the stated vectors,
$q_{12}=1/\sqrt2$, $q_{23}=(1+i)/2$, and $q_{31}=1/\sqrt2$,
so $\Pi_{123}=(1+i)/4$ and its normalized phase is $e^{i\pi/4}$.
Thus this overlap loop has nonzero phase. The projector satisfies
$\det P_i=0$, which also establishes directly that it is not a unitary
color link. $\square$
:::

:::{prf:theorem} Exact phase quotient of a normalized color state
:label: thm-sm-direct-phase-quotient

Let $S^{2n-1}=\{c\in\mathbb C^n:c^\dagger c=1\}$ and
$\mathcal P_{1,n}=\{P=P^\dagger:P^2=P,\operatorname{Tr}P=1\}$.
The map $c\mapsto cc^\dagger$ induces a homeomorphism

$$
S^{2n-1}/U(1)\ \cong\ \mathcal P_{1,n},
$$

where $U(1)$ acts by $c\mapsto e^{i\alpha}c$. On the chart $P_{aa}>0$,
an explicit representative is $c^{[a]}=Pe_a/\sqrt{P_{aa}}$.
Consequently the projector triangle in
{prf:ref}`prop-sm-direct-triangle-projectors` is a function of three
exact phase-quotient coordinates. At $n=3$ this is the phase quotient
of the direct color field; at $n=2$ it is that of a normalized doublet.
:::

:::{prf:proof}
The projector calculation above shows the map takes values in
$\mathcal P_{1,n}$. Each Hermitian idempotent has eigenvalues zero or one,
so trace one gives a one-dimensional image. Choosing a unit vector $c$
in that image gives $P=cc^\dagger$, proving surjectivity.
If $cc^\dagger=dd^\dagger$, applying both sides to $d$ gives
$d=c(c^\dagger d)$. Norms imply $|c^\dagger d|=1$, so
$d=e^{i\alpha}c$. Conversely multiplication by a unit phase preserves
$cc^\dagger$. This proves the fibers are precisely the $U(1)$ orbits.

The chart formula has norm one because
$\|Pe_a\|^2=e_a^\dagger P^\dagger Pe_a=P_{aa}$.
Writing $P=cc^\dagger$ gives
$Pe_a=c\overline c_a$ and thus
$c^{[a]}=c\overline c_a/|c_a|$. It satisfies
$c^{[a]}(c^{[a]})^\dagger=P$ and has positive real $a$th component.
The charts cover the projector space since $\sum_aP_{aa}=1$.
The map from the compact sphere quotient to the Hausdorff projector space
is a continuous bijection; the closed-set argument used in
{prf:ref}`thm-sm-direct-orbit-isomorphism` proves it is a homeomorphism.
$\square$
:::

:::{prf:proposition} Parity of the direct standard pair channels
:label: prop-sm-direct-parity

Suppose spatial inversion acts on the actual recorded ingredients as
$v_i\mapsto-v_i$, $F_i^{\mathrm{visc}}\mapsto-F_i^{\mathrm{visc}}$,
and $r_{ij}\mapsto-r_{ij}$, with reference scales fixed. Then

$$
c_i\mapsto-\overline{c_i},\qquad
q_{ij}\mapsto\overline{q_{ij}},\qquad
b_{ijk}\mapsto-\overline{b_{ijk}},\qquad
\Pi_{ijk}\mapsto\overline{\Pi_{ijk}}.
$$

Hence $\operatorname{Re}q$ is even and $\operatorname{Im}q$ is odd;
$\operatorname{Re}q\,r$ is odd and $\operatorname{Im}q\,r$ is even.
The same identities hold for averaged channels when the companion selection,
masks, and weights transform equivariantly. Parity invariance of expectations
additionally requires invariance of $\mathbb P_{\mathrm{rec}}$.
These parity identities leave rotational spin and charge conjugation to
their separately specified transformations.
:::

:::{prf:proof}
Write $\kappa=m\ell_0/\hbar_{\mathrm{eff}}$. Componentwise,

$$
(c_i')^a=\frac{-F_i^a}{\|-F_i\|}e^{i\kappa(-v_i^a)}
=-\overline{\frac{F_i^a}{\|F_i\|}e^{i\kappa v_i^a}}
=-\overline{c_i^a}.
$$

The pair becomes
$q'_{ij}=\sum_a\overline{(-\overline{c_i^a})}
(-\overline{c_j^a})=\sum_ac_i^a\overline{c_j^a}=\overline{q_{ij}}$.
There are three column minus signs in a determinant, giving
$b'_{ijk}=(-1)^3\det[\overline c_i,\overline c_j,\overline c_k]
=-\overline{b_{ijk}}$. Multiplying the three conjugated pair factors
gives $\Pi'_{ijk}=\overline{\Pi_{ijk}}$.

For $q=x+iy$, conjugation preserves $x$ and negates $y$.
Consequently $(\operatorname{Re}q\,r)'=x(-r)=-xr$, whereas
$(\operatorname{Im}q\,r)'=(-y)(-r)=yr$. If a channel has sign
$s\in\{-1,1\}$ and its weights and mask are unchanged by the relabeling
of inverted pairs, its average transforms as
$\sum_Iw_Im_I(sO_I)/\sum_Iw_Im_I=s\mathcal A_t(O)$.
The zero-denominator convention obeys the same identity.
Finally, for the parity map $\mathcal P$ and an invariant record law,
$\mathbb EO=\mathbb E(O\circ\mathcal P)=s\mathbb EO$.
In particular an integrable odd channel has zero expectation.
$\square$
:::

### Companion doublets without Dirac matrices

:::{prf:definition} Direct companion amplitudes and two-hop doublets
:label: def-sm-direct-companion-doublet

At a fixed frame let $k(i)$ be the cloning companion and let $D_i$ be its
configured algorithmic distance. For positive numerical scales
$\ell_c,h_S,\varepsilon_{\mathrm{clone}}$, set

$$
\theta_i=\frac{F_{k(i)}-F_i}
 {(|F_i|+\varepsilon_{\mathrm{clone}})h_S},\qquad
a_i=\exp[-D_i^2/(4\ell_c^2)]e^{i\theta_i},\qquad
d_i=\begin{pmatrix}a_i\\a_{k(i)}\end{pmatrix}.
$$

The second entry uses the companion's own next companion. It equals a
reverse-pair amplitude only when the companion map and weights give that
identity. With no distance weighting take the exponential amplitude to be
one. Define $z_i=d_i/\|d_i\|$ on nonzero doublets.

In the standard mode of
`src/fragile/physics/operators/electroweak_operators.py`, `su2_component`
uses $a_i$, `su2_doublet` uses $a_i+a_{k(i)}$, and `su2_doublet_diff` uses
$a_i-a_{k(i)}$, followed by masked frame averaging. The code uses the same
numerical `epsilon_clone` as the denominator regularizer and the spatial
amplitude width in this operator path; the displayed $\ell_c$ separates
these roles for dimensional bookkeeping. Equality of their numerical values
is an implementation convention in its chosen units. The normalized $z_i$
and its determinant contractions below are additional mathematical
observables; the current scalar channel names do not assert their computation.

The diversity amplitude is constructed analogously with the distance
companion, its configured bandwidth, and the fitness-difference phase.
All exponents require the stated dimensionless normalization of the scores
and fitness variables. These are observable definitions at a fixed record.
:::

:::{div} feynman-prose
Freeze the inputs just before drawing a companion. Each eligible choice
now has a definite amplitude: distance sets its magnitude and the fitness
difference sets its phase. If two choices have different distances, their
amplitudes cannot coincide, whatever the phases. With positive probabilities
for both choices, the draw therefore produces a fluctuating component.

The variance formula below measures this directly by comparing pairs of
possible outcomes. Its lower bound needs only those two choices and their
actual probabilities. For the two-hop doublet, use the full joint companion
assignment: the components can depend on one another, but the squared
doublet fluctuation still includes the first component's variance. These
are fluctuations before the later masked frame average.
:::

:::{prf:theorem} Nontrivial doublet fluctuations from the actual companion draw
:label: thm-sm-native-doublet-fluctuations

At the actual cloning-companion draw, condition on the complete current
state and the already evaluated fitness inputs, denoting this information
by $\mathcal G$. The companion law is the algorithmic law of
{prf:ref}`def-fg-soft-companion-kernel`; write
$p_{ij}=\mathbb P(K_i=j\mid\mathcal G)$. For the standard distance-weighted
readout in {prf:ref}`def-sm-direct-companion-doublet`, all quantities

(eq-fg-sm-u1)=
$$
r_{ij}=\exp[-D_{ij}^2/(4\ell_c^2)],\quad
\vartheta_{ij}=\frac{F_j-F_i}{(|F_i|+\varepsilon_{\mathrm{clone}})h_S},
\quad a_{ij}=r_{ij}e^{i\vartheta_{ij}},\quad a_i=a_{iK_i}
\tag{SM.U1}
$$

are therefore evaluated directly from this draw and its recorded inputs.
The conditional complex variance, defined using squared modulus, is

(eq-fg-sm-u2)=
$$
\begin{aligned}
\mathbb E[a_i\mid\mathcal G]&=\sum_jp_{ij}a_{ij},\\
\operatorname{Var}(a_i\mid\mathcal G)
 &=\frac12\sum_{j,k}p_{ij}p_{ik}|a_{ij}-a_{ik}|^2\\
 &\ge p_{ij}p_{ik}(r_{ij}-r_{ik})^2
 \qquad(j\ne k).
\end{aligned}
\tag{SM.U2}
$$

In particular every state with two eligible companions at different
finite algorithmic distances has strictly positive conditional component
variance. This strict inequality uses the positive Gaussian companion
weights of the actual finite companion draw; no equilibrium law is needed.

Retain the actual joint companion assignment $\mathbf K$ and form the
two-hop doublet $d_i=(a_i,a_{K_i})^{\mathsf T}$ as in the existing readout.
For each complete assignment $\mathbf k$, let $d_i(\mathbf k)$ denote
its evaluated doublet and let $p(\mathbf k\mid\mathcal G)$ be its actual
joint probability. Then

(eq-fg-sm-u3)=
$$
\begin{aligned}
\mathbb E\!\left[\|d_i-\mathbb E[d_i\mid\mathcal G]\|^2
                         \mid\mathcal G\right]
 &=\frac12\sum_{\mathbf k,\mathbf l}
 p(\mathbf k\mid\mathcal G)p(\mathbf l\mid\mathcal G)
                 \|d_i(\mathbf k)-d_i(\mathbf l)\|^2\\
 &\ge\operatorname{Var}(a_i\mid\mathcal G).
\end{aligned}
\tag{SM.U3}
$$

Thus the recorded doublet has nontrivial stochastic fluctuations already
at the companion substep on these states. Its Hermitian and alternating
contractions are those of {prf:ref}`thm-sm-direct-su2-invariants`.

*Proof.* Conditional on $\mathcal G$, the finite set of values $a_{ij}$
is deterministic. Its conditional expectation and second moment are
$\sum_jp_{ij}a_{ij}$ and $\sum_jp_{ij}|a_{ij}|^2$. Expanding the double sum,

$$
\begin{aligned}
\frac12\sum_{j,k}p_{ij}p_{ik}|a_{ij}-a_{ik}|^2
 &=\sum_jp_{ij}|a_{ij}|^2
    -\operatorname{Re}\sum_{j,k}p_{ij}p_{ik}a_{ij}\overline{a_{ik}}\\
 &=\sum_jp_{ij}|a_{ij}|^2-\left|\sum_jp_{ij}a_{ij}\right|^2.
\end{aligned}
$$

The two terms indexed by $(j,k)$ and $(k,j)$ together contribute
$p_{ij}p_{ik}|a_{ij}-a_{ik}|^2$. The reverse triangle inequality bounds
this below by $p_{ij}p_{ik}(r_{ij}-r_{ik})^2$. Gaussian selection assigns
positive probability to every eligible finite-distance companion, and
$D\mapsto\exp[-D^2/(4\ell_c^2)]$ is strictly decreasing for $D\ge0$.
This proves strict positivity at the stated actual states, without any
restriction on their fitness phases.

Apply the same expansion with the Hermitian norm on $\mathbb C^2$ to
the finite joint companion law. This proves the equality in
{ref}`(SM.U3) <eq-fg-sm-u3>` without independence of the components or
of the two hops. Squared norm is the sum of the two component squared
moduli, so its conditional variance dominates that of the first
component, proving the inequality.

For an actual law of $\mathcal G$, integration retains the explicit bound
$\mathbb E[p_{ij}p_{ik}(r_{ij}-r_{ik})^2]$. It is strictly positive exactly
when the nonnegative integrand is positive on a set of positive measure.
This is the support calculation for the same recorded law, rather than
a substitution of a separate equilibrium measure. It also gives a
lower bound on unconditional doublet variance by conditional variance
decomposition. For the mode without distance weighting, set $r_{ij}=1$:
the exact variance identity remains valid, and distinct phases modulo
$2\pi$ supply its nonzero terms. The distance lower bound itself then
vanishes. Finally, masked frame averaging is a subsequent observable
map; its possible cancellations do not alter the pre-average doublet
variance calculated here.

For the actual completed-record law, let
$f=a_i-\mathbb E a_i\in L^2_0(P)$. The established record CAR
construction gives

$$
\|a^\dagger(f)\Omega\|^2
=\langle f,f\rangle
=\operatorname{Var}_P(a_i)
\ge\mathbb E_P\operatorname{Var}(a_i\mid\mathcal G).
$$

Thus the positive companion-variance calculation gives a nonzero vector
in the fermionic representation of this same record law. Under the
stationary completed-state realization, its ordered time correlations
are the native CAR regression of
{prf:ref}`thm-lqft-record-car-channel`, with the full algorithmic $P$.
This connects the computed fluctuation to the existing quantum
representation without changing the update. $\square$
:::


:::{prf:theorem} Sum and difference channels as exact doublet coordinates
:label: thm-sm-direct-doublet-readout-isomorphism

Before frame averaging, let $s_i^+=a_i+a_{k(i)}$ and
$s_i^-=a_i-a_{k(i)}$. The map $d_i\mapsto s_i=(s_i^+,s_i^-)^{\mathsf T}$
is a complex-linear isomorphism, with

$$
d_i=\frac12\begin{pmatrix}s_i^++s_i^-\\s_i^+-s_i^-\end{pmatrix},
\qquad
\|d_i\|^2=\frac{|s_i^+|^2+|s_i^-|^2}{2}.
$$

Writing $s_i=Hd_i$ with
$H=\left(\begin{smallmatrix}1&1\\1&-1\end{smallmatrix}\right)$,
the map $\mathcal U=H/\sqrt2$ is unitary. It intertwines the defining
$SU(2)$ representation with $B\mapsto\mathcal U B\mathcal U^\dagger$.
Thus retaining both complex readouts retains the full unaveraged doublet.
Retaining $z_i=d_i/\|d_i\|$ also requires the radius $\|d_i\|$ to recover
the original doublet. These are coordinate identities for the ambient
doublet representation; whether a transformed array is generated by the
fixed companion dynamics is determined by
{prf:ref}`prop-sm-direct-law-symmetry`.
:::

:::{prf:proof}
The matrix products are $H^\dagger=H$ and $H^2=2I$, so
$H^{-1}=H/2$ and $\mathcal U^\dagger\mathcal U=I$.
Multiplying $H^{-1}s_i$ gives the displayed inverse. Expanding the
absolute squares gives

$$
|a+b|^2+|a-b|^2
=(|a|^2+|b|^2+2\operatorname{Re}(\overline ab))
 +( |a|^2+|b|^2-2\operatorname{Re}(\overline ab))
=2(|a|^2+|b|^2).
$$

If $d'_i=Bd_i$ then
$s'_i=HBH^{-1}s_i=\mathcal U B\mathcal U^\dagger s_i$.
For $B\in SU(2)$ its conjugate $B_s=\mathcal U B\mathcal U^\dagger$
satisfies

$$
B_s^\dagger B_s=\mathcal U B^\dagger B\mathcal U^\dagger=I,
\qquad
\det B_s=\det\mathcal U\det B\overline{\det\mathcal U}=1.
$$

The inverse conjugation is $B=\mathcal U^\dagger B_s\mathcal U$,
proving representation equivalence in both directions. Finally
$d_i=\|d_i\|z_i$ gives the normalized-coordinate inverse when the
radius is retained. For an average with two independently variable
readouts of positive weights $w_1,w_2$, replacing
$s_1$ by $s_1+\delta$ and $s_2$ by $s_2-(w_1/w_2)\delta$ leaves the
average fixed, since $w_1\delta-w_2(w_1/w_2)\delta=0$.
Invertibility after averaging therefore requires injectivity on the actual
constrained sample space; it does not follow from the per-sample inverse. $\square$
:::

:::{prf:theorem} SU(2) from Hermitian and alternating doublet contractions
:label: thm-sm-direct-su2-invariants

For doublets $z,w\in\mathbb C^2$, set

$$
h(z,w)=z^\dagger w,\qquad
e(z,w)=z^{\mathsf T}\varepsilon w=z_1w_2-z_2w_1,
\qquad
\varepsilon=\begin{pmatrix}0&1\\-1&0\end{pmatrix}.
$$

The complex-linear transformations preserving both contractions for every
$z,w$ are precisely $SU(2)$. For this action the two contractions and all
products of them are invariant under a common frame change. In addition,

$$
|h(z,w)|^2+|e(z,w)|^2=\|z\|^2\|w\|^2.
$$

This gives an exact internal $SU(2)$ observable algebra without Dirac
matrices. The alternating contraction is bilinear in commuting numerical
fields and vanishes at $w=z$; this identity imposes no occupation-number
constraint on the walkers.

The implemented components and scalar readouts
$a_i\pm a_{k(i)}=(1,\pm1)d_i$ are basis dependent. Hermitian contractions
alone select $U(2)$; retaining only $|e|$ also leaves the common determinant
phase undetermined. The full complex alternating form specifies the
determinant-one structure.
:::

:::{prf:proof}
**The stabilizer of the two forms.** Write
$B=\left(\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\right)$.
Hermitian invariance for all $z,w$ gives
$z^\dagger(B^\dagger B-I)w=0$; basis pairs show $B^\dagger B=I$.
For the alternating form the complete multiplication is

$$
B^{\mathsf T}\varepsilon B
=\begin{pmatrix}a&c\\b&d\end{pmatrix}
 \begin{pmatrix}c&d\\-a&-b\end{pmatrix}
=\begin{pmatrix}0&ad-bc\\bc-ad&0\end{pmatrix}
=(\det B)\varepsilon.
$$

Preservation at $z=e_1,w=e_2$ gives $\det B=1$. Conversely these two
matrix identities give
$h(Bz,Bw)=h(z,w)$ and $e(Bz,Bw)=e(z,w)$ for every pair.
The preserving group is therefore exactly $SU(2)$.

**The norm identity.** Expand both absolute squares:

$$
\begin{aligned}
|h(z,w)|^2
 &=|z_1|^2|w_1|^2+|z_2|^2|w_2|^2
   +2\operatorname{Re}(\overline z_1z_2w_1\overline w_2),\\
|e(z,w)|^2
 &=|z_1|^2|w_2|^2+|z_2|^2|w_1|^2
   -2\operatorname{Re}(z_1\overline z_2w_2\overline w_1).
\end{aligned}
$$

The two arguments of $\operatorname{Re}$ are conjugates and thus have
the same real part. Their terms cancel. The four remaining terms are
$(|z_1|^2+|z_2|^2)(|w_1|^2+|w_2|^2)$.

For basis dependence take $d=(1,0)^{\mathsf T}$ and
$B=2^{-1/2}\left(\begin{smallmatrix}1&-1\\1&1\end{smallmatrix}\right)$.
The sum of components changes from $1$ to $\sqrt2$. A simultaneous
transformation of a readout covector $r\mapsto rB^{-1}$ would preserve
$rd$, whereas the implemented fixed covector keeps the recorded component
convention. The identity $e(z,z)=0$ follows from commutativity of the two
scalar components. $\square$
:::

:::{div} feynman-prose
Follow the two routes around one recorded interaction triangle. In temporal
gauge its CST transport is the identity, leaving the comparison between the
IA matrix $A$ and the IG matrix $G$. The Wilson defect measures their
mismatch as a squared matrix distance. It vanishes exactly when the two
transports agree, and its expectation measures that mismatch across the
attributed records.

To evaluate this observable, retain the transport matrices assigned by the
Fractal Set's edge operations. Their ordered product determines the triangle
holonomy. For adjacent triangles, retain the matrix products before taking
the trace, so that their relative transport is included in the larger loop.
:::

:::{prf:proposition} Attribution holonomy and its exact nonflatness observable
:label: prop-sm-attribution-holonomy-defect

Use the Fractal Set attribution connection of
{prf:ref}`def-fractal-set-gauge-connection`, with its prescribed edge
orientations and temporal gauge. Write $A=U^{(2)}_{\mathrm{IA}}$ and
$G=U^{(2)}_{\mathrm{IG}}$ for the two transports in one recorded
interaction triangle. Its holonomy and Wilson defect satisfy

(eq-fg-sm-g9)=
$$
H_\triangle=AG^\dagger,\qquad
w_\triangle=1-\tfrac12\operatorname{Re}\operatorname{Tr}H_\triangle
             =\tfrac14\|A-G\|_{\mathrm F}^2\in[0,2].
\tag{SM.G9}
$$

Consequently the interaction triangle has nonidentity holonomy exactly
when its IA and IG transports differ. The CST, IG, and IA edge operations
of the Fractal Set determine the transports used in this formula.

For any law of the existing attributed records,

(eq-fg-sm-g10)=
$$
\mathbb E w_\triangle
 =\tfrac14\mathbb E\|U^{(2)}_{\mathrm{IA}}
                            -U^{(2)}_{\mathrm{IG}}\|_{\mathrm F}^2,
\qquad
\mathbb E w_\triangle>0
\ \Longleftrightarrow\
\mathbb P(U^{(2)}_{\mathrm{IA}}\ne U^{(2)}_{\mathrm{IG}})>0.
\tag{SM.G10}
$$

*Proof.* The triangle formula is
{prf:ref}`def-fractal-set-wilson-loop`. Expanding the Frobenius norm gives

$$
\begin{aligned}
\|A-G\|_{\mathrm F}^2
 &=\operatorname{Tr}[(A-G)(A^\dagger-G^\dagger)]\\
 &=4-\operatorname{Tr}(AG^\dagger)-\operatorname{Tr}(GA^\dagger)
 =4-2\operatorname{Re}\operatorname{Tr}(AG^\dagger).
\end{aligned}
$$

The eigenvalues of $AG^\dagger\in SU(2)$ are $e^{i\theta},e^{-i\theta}$,
so $w_\triangle=1-\cos\theta\in[0,2]$. Also $AG^\dagger=I$ exactly
when $A=G$. Positivity and boundedness prove
{ref}`(SM.G10) <eq-fg-sm-g10>`. Under a change of vertex frames each
edge transforms at its two ends and the ordered triangle product
transforms by conjugation at its basepoint. Its trace and hence
{ref}`(SM.G9) <eq-fg-sm-g9>` are unchanged. The simplification $H=AG^\dagger$
uses temporal gauge; after a general time-dependent frame change one
retains the CST factor in the full triangle product.

For adjacent triangles, the ordered multiplication and basepoint
conjugation are exactly
{prf:ref}`prop-fractal-set-wilson-factorization`; taking two separate
traces before multiplication would lose this information. When these
edge matrices are evaluated by the recorded attribution rule, retaining
them in the descriptor map makes their action and all moments instances
of {prf:ref}`thm-sm-effective-recorded-gauge-dynamics` and
{prf:ref}`cor-sm-recorded-gauge-generating-functional`. The expectation in {ref}`(SM.G10) <eq-fg-sm-g10>` is evaluated
by inserting the CST, IG, and IA attribution matrices from their recorded
update rule into the complete descriptor likelihood. $\square$
:::





### Observable symmetry, dynamics, and quantum interpretation

:::{prf:proposition} Local frame covariance and symmetry of the record law
:label: prop-sm-direct-law-symmetry

The preceding invariance statements concern common internal transformations
of the descriptor arrays. Under independent color frame changes,

$$
c_i^\dagger c_j\mapsto c_i^\dagger A_i^\dagger A_jc_j.
$$

An invariant comparison of different local fibers is obtained from a
specified link $U_{ij}\mapsto A_iU_{ij}A_j^{-1}$ as
$c_i^\dagger U_{ij}c_j$. A determinant uses three vectors transported
to one common fiber. Doublet contractions have the same requirement with
the $SU(2)$ representation. Fixed common coordinates suffice for the
direct recorded observables; independent local-frame interpretations use
these additional comparisons.

Suppose a group acts measurably on full records and descriptor histories,
the descriptor map $\mathscr D$ satisfies
$\mathscr D(g\mathcal F)=g\mathscr D(\mathcal F)$, and
$g_*\mathbb P_{\mathrm{rec}}=\mathbb P_{\mathrm{rec}}$. Then the descriptor
law is invariant, and the joint laws of invariant direct observables are
unchanged. A sufficient dynamical condition is equivariance of the complete
particle kernel and invariance of its initial law, with an invariant survival
event whenever conditioning is used. An algebraic invariance of contractions
alone leaves these hypotheses to be verified.
:::

:::{prf:proof}
For the untransported pair,
$q'_{ij}=(A_ic_i)^\dagger(A_jc_j)=c_i^\dagger A_i^\dagger A_jc_j$.
For its transported version, the complete cancellation is

$$
(A_ic_i)^\dagger(A_iU_{ij}A_j^{-1})(A_jc_j)
=c_i^\dagger(A_i^\dagger A_i)U_{ij}(A_j^{-1}A_j)c_j
=c_i^\dagger U_{ij}c_j.
$$

Transporting $c_j,c_k$ to $i$ gives
$\det[c_i,U_{ij}c_j,U_{ik}c_k]$. Its transformed value is
$\det[A_ic_i,A_iU_{ij}c_j,A_iU_{ik}c_k]$, which is unchanged because
$\det A_i=1$. Replacing the triple determinant by the doublet alternating
form uses $B_i^{\mathsf T}\varepsilon B_i=\varepsilon$ in the same way.

For a bounded measurable descriptor test $f$, equivariance and law invariance
give the integral calculation

$$
\begin{aligned}
\int f(gx)\,d(\mathscr D_*\mathbb P_{\mathrm{rec}})(x)
&=\int f(g\mathscr D(\mathcal F))\,d\mathbb P_{\mathrm{rec}}(\mathcal F)\\
&=\int f(\mathscr D(g\mathcal F))\,d\mathbb P_{\mathrm{rec}}(\mathcal F)\\
&=\int f(\mathscr D(\mathcal F))\,d\mathbb P_{\mathrm{rec}}(\mathcal F).
\end{aligned}
$$

For a Markov kernel with $P(gs,gA)=P(s,A)$ and invariant initial law
$\mu_0$, the law after one step satisfies

$$
\mu_1(gA)=\int P(s,gA)d\mu_0(s)
=\int P(gr,gA)d\mu_0(r)
=\int P(r,A)d\mu_0(r)=\mu_1(A).
$$

Applying this change of variables successively in
$\mu_0(ds_0)P(s_0,ds_1)\cdots P(s_{k-1},ds_k)$ proves invariance
of every finite path law. If $E$ is an invariant event with positive
probability, then
$\mathbb P(gA\mid E)=\mathbb P(g(A\cap E))/\mathbb P(E)
=\mathbb P(A\mid E)$, proving the conditioned statement.
$\square$
:::

:::{prf:definition} Correlation channels of the direct law
:label: def-sm-direct-correlations

For the specified law and finite second moments, define the connected
temporal autocorrelator

$$
C_O(t,s)=\mathbb E[(\overline O(t)-\mathbb E\overline O(t))
                 (O(s)-\mathbb EO(s))].
$$

Here the bar is complex conjugation, and $O$ denotes a chosen frame
observable. A vector channel contracts its component autocorrelators.
Stationarity makes $C_O$ a function of the time difference. A finite-record
FFT with empirical mean subtraction estimates this correlation subject to
its sampling, temporal dependence, and centering errors.
For positive real values at two consecutive lags the log-ratio statistic is
$r_O(t;h)=-h^{-1}\log[C_O(t+h)/C_O(t)]$.

When this same correlator has a positive transfer spectral representation
$C_O(t)=\int e^{-Et}\,d\nu_O(E)$, its decay is a spectral measurement
for that transfer generator. Without this identification $r_O$ remains
a recorded decay statistic. An energy interpretation uses a calibrated
action unit, and a mass interpretation additionally uses the physical time,
momentum channel, and speed convention. Matrix-valued Dirac bilinears are
another possible observable map; their comparison with the direct channels
requires a common law and a verified common spectral sector.
:::

### Exact invariant coordinates and transport of the established estimates

:::{prf:theorem} Gram and determinant coordinates for SU(n) orbits
:label: thm-sm-direct-orbit-isomorphism

Let $n\in\{2,3\}$ and $M\ge1$. Write
$X_{n,M}=\{C=(c_1,\ldots,c_M)\in\mathbb C^{n\times M}:\|c_i\|=1\}$.
For each increasing $n$-tuple $I$ of column indices let $B_I=\det C_I$,
and set $G=C^\dagger C$. Define $\mathcal I_{n,M}$ by the finite conditions

$$
G=G^\dagger\succeq0,\quad \operatorname{rank}G\le n,\quad G_{ii}=1,
\qquad \overline{B_I}B_J=\det G_{I,J}\quad\hbox{for all }I,J.
$$

When $M<n$ the list of $B_I$ is empty. The invariant map
$q(C)=(G,(B_I)_I)$ induces a homeomorphism

$$
X_{n,M}/SU(n)\ \cong\ \mathcal I_{n,M}.
$$

Consequently pullback is an isometric unital $*$-algebra isomorphism
$C(\mathcal I_{n,M})\cong C(X_{n,M})^{SU(n)}$. At $n=3$ these coordinates
are the full pair and determinant observables; at $n=2$ they are the full
Hermitian and alternating doublet contractions. The result includes
rank-deficient configurations.
:::

:::{prf:proof}
**1. Every configuration satisfies the coordinate equations.** For
$\alpha\in\mathbb C^M$,

$$
\alpha^\dagger G\alpha=\alpha^\dagger C^\dagger C\alpha
=\|C\alpha\|^2\ge0,\qquad G_{ii}=\|c_i\|^2=1.
$$

The rank is at most the number $n$ of rows of $C$. For increasing
$n$-tuples $I,J$, the matrix $G_{I,J}$ has entries
$(G_{I,J})_{ab}=c_{I_a}^\dagger c_{J_b}$, hence

$$
\det G_{I,J}
=\det(C_I^\dagger C_J)
=\overline{\det C_I}\det C_J
=\overline{B_I}B_J.
$$

**2. Construct a representative for every admissible coordinate list.**
Diagonalize $G=W\operatorname{diag}(\lambda_1,\ldots,\lambda_r,0,\ldots)W^\dagger$
with $\lambda_a>0$ and $r=\operatorname{rank}G\le n$.
Define the first $r$ rows of $C_0$ by
$(C_0)_{aj}=\sqrt{\lambda_a}\,\overline{W_{ja}}$, and set its
remaining rows to zero. Then

$$
(C_0^\dagger C_0)_{ij}
=\sum_{a=1}^r\lambda_aW_{ia}\overline{W_{ja}}=G_{ij}.
$$

In particular its columns are unit vectors. If $r<n$, each $n$-minor
vanishes, so $|B_I|^2=\det G_{I,I}=0$ for every $I$.
Thus $C_0$ already represents the prescribed data, including all determinants.
This also covers $M<n$, when there are no $n$-tuples.

If $r=n$, choose $I_0$ with $B^0_{I_0}:=\det(C_0)_{I_0}\ne0$.
The diagonal relation gives $|B_{I_0}|=|B^0_{I_0}|$; therefore
$\zeta=B_{I_0}/B^0_{I_0}$ has unit modulus. For every $J$,

$$
\overline{B_{I_0}}B_J
=\det G_{I_0,J}=\overline{B^0_{I_0}}B^0_J,
\qquad
B_J=\frac{B^0_J}{\overline\zeta}=\zeta B^0_J.
$$

Take $A=\operatorname{diag}(\zeta,1,\ldots,1)$ and $C=AC_0$.
Then $C^\dagger C=C_0^\dagger C_0=G$ and
$\det C_J=\det A\det(C_0)_J=\zeta B^0_J=B_J$ for every $J$.
This constructs a representative on every rank stratum.

**3. Equal coordinates give the same SU(n) orbit.** Suppose $C,D$ have
the same $G$ and $B$. Define $T$ on the column span of $C$ by
$T(C\alpha)=D\alpha$. If $C\alpha=C\beta$, then

$$
\|D(\alpha-\beta)\|^2
=(\alpha-\beta)^\dagger G(\alpha-\beta)
=\|C(\alpha-\beta)\|^2=0.
$$

Thus $T$ is well-defined. For arbitrary $\alpha,\beta$,
$\langle D\alpha,D\beta\rangle=\alpha^\dagger G\beta
=\langle C\alpha,C\beta\rangle$, so $T$ is an isometry onto the
column span of $D$. Extend orthonormal bases of these two spans to
orthonormal bases of $\mathbb C^n$ and map the added basis vectors in
order. This produces a unitary $A$ with $D=AC$.

If the rank is $n$, choose an invertible $C_I$. Then

$$
B_I=\det D_I=\det(A)\det C_I=\det(A)B_I,
$$

and $B_I\ne0$ gives $\det A=1$.
If the rank is less than $n$, choose a unit vector $u$ orthogonal to every
column of $D$ and let $\eta=(\det A)^{-1}$. Set

$$
R=I+(\eta-1)uu^\dagger.
$$

The matrix $R$ multiplies $u$ by $\eta$ and fixes $u^\perp$. Hence
$R$ is unitary, $\det R=\eta$, and $RD=D$. It follows that
$\widetilde A=RA$ satisfies $\det\widetilde A=1$ and
$\widetilde AC=D$. Conversely a common $SU(n)$ transformation
preserves the coordinates by the pair and determinant calculations above.
This proves both directions of orbit separation.

**4. Topology and observable algebra.** The space $X_{n,M}$ is a finite
product of unit spheres and is compact. The entries of $q$ are polynomials
in the real and imaginary coordinates, so $q$ is continuous. It descends
to a continuous bijection from the compact quotient $X_{n,M}/SU(n)$ to
the Hausdorff space $\mathcal I_{n,M}$. Images of closed subsets are
compact and therefore closed, proving continuity of the inverse.

For a continuous invariant $F$ define $f(q(C))=F(C)$. Orbit separation
makes $f$ well-defined; the quotient homeomorphism makes it continuous.
Conversely $f\circ q$ is invariant and continuous. Finally,

$$
(fg)\circ q=(f\circ q)(g\circ q),\quad
\overline f\circ q=\overline{f\circ q},\quad
1\circ q=1,\quad
\|f\circ q\|_\infty=\sup_{y\in\mathcal I_{n,M}}|f(y)|.
$$

These identities prove the stated isometric $*$-algebra isomorphism.
$\square$
:::

:::{prf:corollary} Explicit reconstruction on a full-rank anchor chart
:label: cor-sm-direct-anchor-inverse

Suppose $G_{I,I}$ is positive definite for an $n$-tuple $I$. The entries
$G_{I,I}$, $G_{I,j}$ for all $j$, and $B_I$ determine the configuration up
to $SU(n)$. Choose any invertible $C_I$ with
$C_I^\dagger C_I=G_{I,I}$ and $\det C_I=B_I$. Then an inverse representative
is given by

$$
c_j=(C_I^\dagger)^{-1}G_{I,j}.
$$

For fixed $C_I$ the reconstruction error satisfies
$\|\delta c_j\|\le\sigma_{\min}(C_I)^{-1}\|\delta G_{I,j}\|$.
This chart requires a nonzero anchor determinant; the full orbit theorem
covers the lower-rank strata. A sparse companion graph supplies such an
inverse only if it records or reconstructs the required anchor contractions.
:::

:::{prf:proof}
Put $K=G_{I,I}\succ0$ and let $R=K^{1/2}$ be its positive square root.
Then $R^\dagger R=K$ and $\det R=\sqrt{\det K}>0$.
The number $\eta=B_I/\sqrt{\det K}$ has modulus one. With
$A=\operatorname{diag}(\eta,1,\ldots,1)$, set $C_I=AR$. It satisfies

$$
C_I^\dagger C_I=R A^\dagger AR=K,\qquad
\det C_I=\eta\sqrt{\det K}=B_I.
$$

Solving $C_I^\dagger c_j=G_{I,j}$ gives the stated formula.
To check that the reconstructed columns recover the complete Gram matrix,
reorder the anchor indices first and write

$$
G=\begin{pmatrix}K&H\\H^\dagger&J\end{pmatrix}.
$$

The block multiplication is

$$
\begin{pmatrix}I&0\\-H^\dagger K^{-1}&I\end{pmatrix}
\begin{pmatrix}K&H\\H^\dagger&J\end{pmatrix}
\begin{pmatrix}I&-K^{-1}H\\0&I\end{pmatrix}
=\begin{pmatrix}K&0\\0&J-H^\dagger K^{-1}H\end{pmatrix}.
$$

Both triangular factors are invertible. Consequently
$\operatorname{rank}G=n+\operatorname{rank}(J-H^\dagger K^{-1}H)$.
The bound $\operatorname{rank}G\le n$ forces the Schur complement to
have rank zero and hence to vanish. Thus
$G_{jk}=G_{j,I}K^{-1}G_{I,k}$ for every pair of columns. The inverse
formula gives exactly

$$
c_j^\dagger c_k
=G_{j,I}C_I^{-1}(C_I^\dagger)^{-1}G_{I,k}
=G_{j,I}K^{-1}G_{I,k}=G_{jk}.
$$

For any increasing $n$-tuple $J$, the determinant relation with the anchor
gives
$\overline{B_I}\det C_J=\det G_{I,J}=\overline{B_I}B_J$;
cancelling the nonzero anchor proves $\det C_J=B_J$.
For perturbations of the cross entries at fixed anchor,

$$
\delta c_j=(C_I^\dagger)^{-1}\delta G_{I,j},\qquad
\|\delta c_j\|\le\|(C_I^\dagger)^{-1}\|_{\mathrm{op}}
                         \|\delta G_{I,j}\|
=\frac{\|\delta G_{I,j}\|}{\sigma_{\min}(C_I)}.
$$

Uniqueness up to $SU(n)$ follows from orbit separation. $\square$
:::

:::{prf:theorem} Measure and observable isomorphism for invariant coordinates
:label: thm-sm-direct-measure-isomorphism

For any probability law $\mu$ on $X_{n,M}$ let $\nu=q_*\mu$ on
$\mathcal I_{n,M}$. No group invariance of $\mu$ is required. The map

$$
V:L^2(\nu)\longrightarrow L^2(\sigma(q),\mu),\qquad Vf=f\circ q,
$$

is unitary and, for bounded measurable $f$, intertwines multiplication observables:
$VM_fV^{-1}=M_{f\circ q}$. Applying $q$ framewise gives the same
identification of invariant history observables and preserves every
integrable multi-time correlation exactly. The group acts commonly within
each specified frame; comparisons requiring a common frame across times
instead include those times in one descriptor array or record their transports.

The invariant channels formed from a subset of these coordinates, followed by
masking and averaging, are measurable functions of this full representation
and of the recorded auxiliary data. Their law is obtained by a further
pushforward. Invertibility of that additional compression requires its own
injectivity; the full-coordinate theorem does not attribute an inverse to
the displayed channel averages.
Basis-dependent doublet components instead use their complete readout
coordinates in {prf:ref}`thm-sm-direct-doublet-readout-isomorphism`, or
retain the frame needed to evaluate the chosen readout covectors.
:::

:::{prf:proof}
**Isometry and range.** The definition $\nu(A)=\mu(q^{-1}A)$ first gives
$\int f\,d\nu=\int f\circ q\,d\mu$ for indicators, then simple
functions, and then nonnegative functions by monotone convergence.
Applying it to $|f|^2$ and $\overline f g$ yields

$$
\|Vf\|_{L^2(\mu)}^2=\int|f(q(C))|^2d\mu(C)=\int|f(y)|^2d\nu(y),
\quad
\langle Vf,Vg\rangle_\mu=\langle f,g\rangle_\nu.
$$

The range of an isometry from a complete space is closed. It contains
$\mathbf1_{q^{-1}A}=V\mathbf1_A$ for every Borel $A$. Simple functions
of these sets are dense in $L^2(\sigma(q),\mu)$, proving surjectivity
onto that subspace. Thus $V$ is unitary with the specified codomain.

**Multiplication.** For bounded $f$ and $g\in L^2(\nu)$,

$$
(VM_fg)(C)=f(q(C))g(q(C))=(M_{f\circ q}Vg)(C).
$$

Since $V$ is onto, this is the claimed conjugation identity.

**Multi-time law.** For times $t_1,\ldots,t_k$, let
$\mu^{(k)}$ be the actual joint law of $(C(t_1),\ldots,C(t_k))$ and
$\nu^{(k)}=(q,\ldots,q)_*\mu^{(k)}$. No independence is used. For
integrable products,

$$
\int\prod_{a=1}^k f_a(y_a)\,d\nu^{(k)}(y_1,\ldots,y_k)
=\int\prod_{a=1}^k f_a(q(C_a))\,d\mu^{(k)}(C_1,\ldots,C_k).
$$

The identity also holds with any factors conjugated. The one-point means
are equal, so subtracting their products preserves connected correlations
as well. If $a$ denotes recorded auxiliary data, replace $q$ by
$\widetilde q(C,a)=(q(C),a)$ throughout; the same integral equalities
prove the claims with masks, distances, scores, and weights included.
$\square$
:::

:::{prf:proposition} Criterion for an induced Markov evolution
:label: prop-sm-direct-markov-intertwining

Let $P$ be the actual conservative transition kernel on a standard Borel
state space and $q$ a measurable descriptor map. A transition kernel
$\overline P$ on descriptor values intertwines the dynamics precisely when

$$
P(s,q^{-1}A)=\overline P(q(s),A)
$$

for every state and measurable descriptor set $A$. In that case
$P(f\circ q)=(\overline Pf)\circ q$ and the identity iterates. If the
actual invariant law is $\mu$, then $q_*\mu$ is invariant for
$\overline P$. For a strongly continuous semigroup, a reducing descriptor subspace in $L^2(\mu)$
therefore carries the unitarily conjugate semigroup, its generator, and its
spectrum. A self-adjoint positive transfer representation on that subspace
is preserved by the unitary map of
{prf:ref}`thm-sm-direct-measure-isomorphism`.

For the compact orbit map of {prf:ref}`thm-sm-direct-orbit-isomorphism`, a
group-equivariant kernel gives such an induced kernel: its pushforward
transition probabilities are constant on each orbit. A direct history law
and its correlations remain defined even when a selected compressed
descriptor is not Markovian.
:::

:::{prf:proof}
For $f=\mathbf1_A$ the intertwining equation is exactly the displayed
kernel equation. Linearity extends it to simple functions, and bounded
convergence extends it to bounded measurable $f$. Conversely the function
equation applied to every indicator gives the kernel equation. Iteration
gives, for every integer $k\ge1$,

$$
P^{k+1}(f\circ q)
=P((\overline P^kf)\circ q)
=(\overline P^{k+1}f)\circ q.
$$

Writing $\nu=q_*\mu$, stationarity is transported by

$$
\int\overline Pf\,d\nu
=\int(\overline Pf)\circ q\,d\mu
=\int P(f\circ q)d\mu
=\int f\circ q\,d\mu=\int f\,d\nu.
$$

For a continuous-time semigroup restricted to a reducing descriptor space
$\mathcal H_q$, let $V$ be the unitary onto $\mathcal H_q$ and set
$\overline P_t=V^{-1}P_t|_{\mathcal H_q}V$. Then

$$
\overline P_t\overline P_s
=V^{-1}P_tVV^{-1}P_sV
=V^{-1}P_{t+s}V=\overline P_{t+s}.
$$

If $L$ is its restricted generator, difference quotients show
$\overline L=V^{-1}LV$ with domain $V^{-1}\operatorname{Dom}L$.
For every resolvent parameter $z$,
$(z-\overline L)^{-1}=V^{-1}(z-L)^{-1}V$, proving equality of spectra.
Likewise $\overline P_t^*=V^{-1}P_t^*V$, preserving self-adjointness.
For a nonnegative transfer generator $H$,
$\langle f,V^{-1}HVf\rangle=\langle Vf,HVf\rangle\ge0$ on its
transported domain.

For equivariant kernels on $X_{n,M}$,
$q^{-1}A$ is invariant under each group element, and hence
$P(gs,q^{-1}A)=P(s,g^{-1}q^{-1}A)=P(s,q^{-1}A)$.
To see measurability of the resulting quotient kernel explicitly, choose
the first lexicographic maximal independent anchor in $G$. On its
fixed-rank stratum use the positive square-root construction of
{prf:ref}`cor-sm-direct-anchor-inverse` in that rank, padding zero rows.
At full rank apply the determinant-phase adjustment already proved above.
Ranks and anchor choices are Borel conditions on minors, and square roots
and inverses are continuous on each positive-definite anchor chart.
This defines a Borel representative $s(y)$ with $q(s(y))=y$.
Then $\overline P(y,A)=P(s(y),q^{-1}A)$ is a measurable probability
kernel independent of the chosen representative. $\square$
:::

:::{div} feynman-prose
Suppose you display just one channel from the swarm. Several complete states
can give the same displayed value. If their remaining coordinates affect the
next readout, averaging those coordinates away after every step can change the
predictions several steps ahead.

We can keep that effect exactly. Start with an observable of the displayed
channel and apply the actual transition operator. Its prediction may now
depend on more of the swarm state. The projection $\Pi$ keeps the conditional
average visible through the channel; $R$ keeps the remaining dependence.
The four blocks below track how predictions pass between these two parts.
Eliminating the second part gives an explicit memory sum built from the
same recorded kernel. This computes the projected evolution for the chosen
channel, including a masked state readout, without assuming that its present
value suffices to predict its future.
:::

:::{prf:theorem} Exact recorded-channel dynamics with its eliminated-coordinate memory
:label: thm-sm-direct-channel-memory

Use the actual conservative stationary kernel $P$ represented in
{prf:ref}`thm-sm-instantiated-record-transition`, and any recorded
state descriptor $q$, including the invariant coordinates and their
masked channel readouts. On $\mathcal H=L^2(\pi)$ let
$\Pi f=\mathbb E_\pi[f\mid\sigma(q)]$,
$R=I-\Pi$, and $\mathcal H_q=\operatorname{Ran}\Pi$. The existing
unitary $V:L^2(q_*\pi)\to\mathcal H_q$ identifies this subspace with
the descriptor law. Decompose the *same* recorded kernel as

(eq-fg-sm-m1)=
$$
P=
\begin{pmatrix}\mathsf A&\mathsf B\\
                \mathsf C&\mathsf D\end{pmatrix},
\quad
\mathsf A=\Pi P|_{\mathcal H_q},\quad
\mathsf B=\Pi P|_{\operatorname{Ran}R},\quad
\mathsf C=RP|_{\mathcal H_q},\quad
\mathsf D=RP|_{\operatorname{Ran}R}.
\tag{SM.M1}
$$

All blocks are contractions. For $T_n=\Pi P^n|_{\mathcal H_q}$,
the exact two-time channel transition obeys

(eq-fg-sm-m2)=
$$
\begin{aligned}
T_0&=I_{\mathcal H_q},\\
T_{n+1}
 &=\mathsf A T_n+
   \sum_{j=0}^{n-1}\mathsf B\mathsf D^{\,n-1-j}\mathsf C T_j,
 \qquad n\ge0.
\end{aligned}
\tag{SM.M2}
$$

The sum is empty at $n=0$. Every block is an operator of the complete
algorithm; the memory terms specify the effect of the discarded state
coordinates. In particular,

(eq-fg-sm-m3)=
$$
T_2-\mathsf A^2=\Pi PRP|_{\mathcal H_q}
                 =\mathsf B\mathsf C.
\tag{SM.M3}
$$

For $|z|<1$ the norm-convergent generating function is

(eq-fg-sm-m4)=
$$
\sum_{n=0}^\infty z^nT_n
=\left[I-z\mathsf A
 -z^2\mathsf B(I-z\mathsf D)^{-1}\mathsf C\right]^{-1}.
\tag{SM.M4}
$$

The complete hierarchy of bounded channel insertions is obtained by
the same block multiplication, including its hidden-state blocks.
When $\mathsf C=0$, the represented channel subspace is invariant,
the memory vanishes, and its exact transition is $\mathsf A^n$.
This is the stationary $L^2$ realization of the already stated
intertwining criterion. For the complete Fractal Set encoding $\Pi=I$,
so this reduction recovers the exact kernel conjugation.
:::

:::{prf:proof}
**Conditional law and decomposition.** Conditional expectation is an
orthogonal projection, and the stationary $P$ is a contraction.
This proves the block bounds. For a bounded descriptor test $f$,

$$
(V^{-1}T_nVf)(q(S_0))
=\mathbb E_\pi[f(q(S_n))\mid q(S_0)].
$$

The equality follows by conditioning first on $S_0$ and then on $q(S_0)$.
Thus $T_n$ is exactly the recorded two-time conditional operator.
It also has a probability kernel on the standard Borel descriptor space.
In particular $\mathsf A$ is its one-step operator; the following
calculation determines whether its powers suffice.

For $f\in\mathcal H_q$, put $x_n=\Pi P^nf$ and $y_n=RP^nf$.
Their initial values are $x_0=f$, $y_0=0$. Block multiplication gives

$$
x_{n+1}=\mathsf A x_n+\mathsf B y_n,\qquad
y_{n+1}=\mathsf C x_n+\mathsf D y_n.
$$

Iterating the second equation gives
$y_n=\sum_{j=0}^{n-1}\mathsf D^{\,n-1-j}\mathsf Cx_j$.
Insert it into the first equation to prove
{ref}`(SM.M2) <eq-fg-sm-m2>`, and take $n=1$ to obtain
{ref}`(SM.M3) <eq-fg-sm-m3>`.

**Resolvent calculation.** Since $\|P\|\le1$ and $|z|<1$,
$(I-zP)^{-1}=\sum_{n\ge0}z^nP^n$ in operator norm.
To solve $(I-zP)(x,y)=(f,0)$, its second block gives
$y=z(I-z\mathsf D)^{-1}\mathsf Cx$.
Its first block then reads

$$
\left[I-z\mathsf A
 -z^2\mathsf B(I-z\mathsf D)^{-1}\mathsf C\right]x=f.
$$

Both full and lower-block resolvents exist by their Neumann series;
block elimination proves that the bracketed operator is invertible.
Taking the resolved component proves
{ref}`(SM.M4) <eq-fg-sm-m4>`.

**All channel correlations.** Multiplication by a bounded $q$-measurable
observable $a$ commutes with $\Pi$:
$\Pi(af)=a\Pi f$. It is therefore block diagonal on
$\mathcal H_q\oplus\operatorname{Ran}R$.
The complete formula {ref}`(SM.K4) <eq-fg-sm-k4>` has initial and
terminal vector $1\in\mathcal H_q$. Replace each $P$ by
{ref}`(SM.M1) <eq-fg-sm-m1>` and each insertion by its two diagonal
blocks. Expanding the finite product gives the identical scalar
correlation, including every excursion into and return from
$\operatorname{Ran}R$. This supplies the higher-time correspondence;
two-time conditional operators alone need not determine it.

Finally $\mathsf C=0$ says $P\mathcal H_q\subseteq\mathcal H_q$.
All iterates then stay in that subspace, giving $T_n=\mathsf A^n$
and closure of the inserted products. Encoding all coordinates makes
$R=0$. For a time-scheduled implementation these formulas use the
stationary completed state or its fixed-phase stroboscopic kernel from
{prf:ref}`rem-sm-actual-step-and-clock`.
:::


:::{div} feynman-prose
There is a direct way to retain the information needed for prediction.
Start with the channels you intend to measure, including their masks. Apply
the algorithm's transition to their observables: this gives their expected
values one step later as functions of the current complete state. Keep those
prediction functions as additional coordinates. Include products and repeat,
so that joint readouts and their next-step predictions are retained too.

The following construction carries out this procedure through all stages.
It produces the smallest observable sigma-algebra containing the selected
channels and preserved by the actual transition. That completed descriptor
has an exact stationary Markov law and preserves the original channel
correlations. The construction may require countably many coordinates;
the finite partitions that follow provide its explicit approximations.
:::

:::{prf:theorem} Prediction-complete gauge descriptors from the actual transition
:label: thm-sm-prediction-complete-descriptors

Use the conservative stationary completed-state kernel $P$ and law $\pi$
already represented in {prf:ref}`thm-sm-instantiated-record-transition`
and {prf:ref}`thm-sm-direct-channel-memory`. Begin with the countable
collection of bounded recorded channel coordinates $q_j$ under study,
including their validity masks and the constant $1$. A finite collection
is included. Let $\mathcal A_0$ be their unital algebra over
$\mathbb Q+i\mathbb Q$, with complex conjugates included, and recursively
form the countable algebras

(eq-fg-sm-t1)=
$$
\mathcal A_{r+1}
=\operatorname{alg}_{\mathbb Q+i\mathbb Q}
  (\mathcal A_r\cup P\mathcal A_r\cup\overline{P\mathcal A_r}),
\qquad
\Sigma_{\mathrm{pred}}=\sigma\left(\bigcup_{r\ge0}\mathcal A_r\right).
\tag{SM.T1}
$$

Enumerate that union as $(f_j)_{j\ge1}$ and set
$\widehat q(s)=(f_j(s))_{j\ge1}\in\mathbb C^{\mathbb N}$.
This descriptor is calculated from the original channels and the actual
algorithmic kernel. Its observable space
$\mathcal H_{\mathrm{pred}}=L^2(\Sigma_{\mathrm{pred}},\pi)$ is
$P$-invariant. With $\widehat\nu=\widehat q_*\pi$ and the pullback
unitary $Vf=f\circ\widehat q$, its exact kernel is

(eq-fg-sm-t2)=
$$
\widehat P=V^{-1}P|_{\mathcal H_{\mathrm{pred}}}V,
\qquad
PV=V\widehat P,\qquad
\widehat\nu\widehat P=\widehat\nu.
\tag{SM.T2}
$$

The process $\widehat q(S_n)$ is Markov under the stationary recorded
law, and every finite correlation of the original channels is unchanged.
Moreover $\Sigma_{\mathrm{pred}}$ is the smallest completed observable
sigma-algebra containing those channels whose bounded functions are
preserved by $P$. The memory term in
{ref}`(SM.M2) <eq-fg-sm-m2>` vanishes on this completed space.
:::

:::{prf:proof}
**1. Close the observable space using the given update.** The algebras
are countable because they use countably many finite rational operations.
All their elements are bounded, since a Markov kernel preserves boundedness.
Their union $\mathcal A$ is an algebra and satisfies $P\mathcal A\subseteq
\mathcal A$. Bounded real functions $f$ measurable for
$\Sigma_{\mathrm{pred}}$ with $Pf$ measurable for that sigma-algebra
form a vector space closed under uniformly bounded pointwise limits:
if $f_n\to f$, dominated convergence in $P(s,ds')$ gives
$Pf_n(s)\to Pf(s)$. The functional monotone-class theorem applied to
the real algebra generated by $\mathcal A$ therefore gives
$P L^\infty(\Sigma_{\mathrm{pred}})\subseteq
L^\infty(\Sigma_{\mathrm{pred}})$. Complexification gives the same
statement for complex functions. Stationarity ensures that $P$ respects
$\pi$-null modifications: for $\pi(A)=0$,
$\int P(s,A)d\pi(s)=\pi(A)=0$.
Finally, truncation and the $L^2(\pi)$ contraction property extend this
invariance to $\mathcal H_{\mathrm{pred}}$.

**2. Identify the actual conditional kernel.** The descriptor target is
standard Borel. Disintegrate the stationary two-time law
$\pi(ds)P(s,ds')$ conditional on $\widehat q(s)$ and push the second
coordinate through $\widehat q$. This gives a probability kernel
$\widehat P(y,dy')$. For bounded descriptor $g$, invariance from step 1
means $P(g\circ\widehat q)=h\circ\widehat q$ for some measurable $h$.
The disintegration identifies $h=\widehat Pg$, proving
{ref}`(SM.T2) <eq-fg-sm-t2>`. Integrating that equality proves invariance
of $\widehat\nu$. For the recorded history $\mathscr F_n$,

$$
\begin{aligned}
\mathbb E[g(\widehat q(S_{n+1}))\mid
             \widehat q(S_0),\ldots,\widehat q(S_n)]
&=\mathbb E[P(g\circ\widehat q)(S_n)\mid
             \widehat q(S_0),\ldots,\widehat q(S_n)]\\
&=\widehat Pg(\widehat q(S_n)).
\end{aligned}
$$

This proves the Markov property directly. Multiplication by any original
channel preserves the completed space. Thus every ordered product of
these multiplications and powers of $P$ stays there. Conjugating that
product by $V$ gives its exact descriptor correlation, including the
original source and terminal vector $1$.

**3. Minimality and memory.** Any completed sigma-algebra containing
$q_j$ and preserved by $P$ on bounded functions contains $\mathcal A_0$;
induction gives every $\mathcal A_r$. It therefore contains
$\Sigma_{\mathrm{pred}}$. Conversely step 1 proves that this sigma-algebra
has the stated property. Its conditional projection satisfies
$(I-\Pi_{\mathrm{pred}})P\Pi_{\mathrm{pred}}=0$, which sets
$\mathsf C=0$ in {ref}`(SM.M1) <eq-fg-sm-m1>` and proves the memory
assertion. This completes the existing channel intertwining criterion by
constructing its invariant space from the recorded update itself.
:::

:::{div} feynman-prose
To turn these predictions into a finite matrix, divide the completed
descriptor values into cells. Each matrix entry is the probability that the
next actual update lands in a destination cell, conditional on starting in
the source cell under the stationary law. A channel is represented by its
average within each cell. These probabilities and averages are quantities
to evaluate from that same recorded law.

Refining the cells and retaining more prediction coordinates gives the
convergence proved below. At finite resolution, iterating the cell matrix
performs an approximation. The theorem shows that each fixed finite sequence
of transitions and observations converges to its recorded correlation as
the partitions become finer; it also carries that limit to the CAR maps.
:::

:::{prf:theorem} Finite transition matrices converging to the completed channel theory
:label: thm-sm-predictive-partition-convergence

For the descriptor in {prf:ref}`thm-sm-prediction-complete-descriptors`,
form nested finite partitions $\mathcal P_M$ by dyadically quantizing the
real and imaginary parts of its first $M$ coordinates, using resolution
$2^{-M}$ and outer tail cells beyond $[-M,M]$. Their generated
sigma-algebras increase to $\Sigma_{\mathrm{pred}}$.
Let $\Pi_M$ be conditional expectation onto this finite sigma-algebra.
For its positive-probability cells $A_a$, define

(eq-fg-sm-t3)=
$$
w_a=\pi(A_a),\qquad
p_{ab}^{(M)}=\frac1{w_a}\int_{A_a}P(s,A_b)d\pi(s),\qquad
b_a^{(M)}=\frac1{w_a}\int_{A_a}b(s)d\pi(s).
\tag{SM.T3}
$$

These are the actual stationary cell probabilities, transition
probabilities, and cell averages of a bounded channel $b$.
The finite kernel has invariant law $(w_a)$ and represents
$P_M=\Pi_M P\Pi_M$. Its bounded insertion is
$B_M=\Pi_M M_b\Pi_M$, represented by the diagonal entries $b_a^{(M)}$.
On $\mathcal H_{\mathrm{pred}}$,

(eq-fg-sm-t4)=
$$
P_M\longrightarrow P|_{\mathcal H_{\mathrm{pred}}},\qquad
B_M\longrightarrow M_b
\quad\text{strongly}.
\tag{SM.T4}
$$

Every fixed finite ordered correlation formed from these finite matrices
therefore converges to the corresponding recorded channel correlation.
The centered contractions also induce convergent CAR maps and convergent
finite ordered CAR regression correlations by the construction in
{prf:ref}`thm-lqft-record-car-channel`.
:::

:::{prf:proof}
**1. Compute the finite matrices.** For a cell-constant function
$f=\sum_bf_b\mathbf1_{A_b}$, conditional averaging gives

$$
(\Pi_M P\Pi_M f)|_{A_a}
=\sum_b\frac{\int_{A_a}P(s,A_b)d\pi(s)}{w_a}f_b.
$$

Nonnegativity and $\sum_bp_{ab}^{(M)}=1$ follow from the kernel.
Stationarity gives
$\sum_aw_ap_{ab}^{(M)}=\int P(s,A_b)d\pi(s)=w_b$.
Applying the same conditional average to $bf$ gives the stated diagonal
insertion. Zero-probability cells contribute zero to all these integrals.
The normalized indicators $\mathbf1_{A_a}/\sqrt{w_a}$ form an
orthonormal basis; in that basis the transition entries are
$\sqrt{w_a}\,p_{ab}^{(M)}/\sqrt{w_b}$.

**2. Establish the strong limit.** The partitions separate all descriptor
coordinates, so the increasing orthogonal projections satisfy
$\Pi_M\to I$ strongly on $\mathcal H_{\mathrm{pred}}$.
For completeness, their range union is dense: indicators of cells generate
the sigma-algebra, and bounded simple approximation and the monotone-class
argument give density in $L^2$. For $f$ in that space,

$$
\begin{aligned}
\|(P_M-P)f\|_2
&\le\|(\Pi_M-I)f\|_2+\|(\Pi_M-I)Pf\|_2,\\
\|(B_M-M_b)f\|_2
&\le\|b\|_\infty\|(\Pi_M-I)f\|_2
                 +\|(\Pi_M-I)bf\|_2.
\end{aligned}
$$

Every term tends to zero. Moreover $\|P_M\|\le1$ and
$\|B_M\|\le\|b\|_\infty$. For any finite list of these factors,
with limits $A_j$, the exact error is

(eq-fg-sm-t5)=
$$
\left(\prod_{j=1}^rA_{j,M}-\prod_{j=1}^rA_j\right)f
=\sum_{j=1}^r\left(\prod_{i<j}A_{i,M}\right)
 (A_{j,M}-A_j)\left(\prod_{i>j}A_i\right)f.
\tag{SM.T5}
$$

Each middle difference acts on a fixed vector, and all preceding factors
are uniformly bounded. Every summand tends to zero. Taking the matrix
element against $1$, which belongs to every partition space, proves the
correlation limit with explicit finite-approximation error terms.

**3. Carry the same limit into the fermionic representation.** On the
centered mode space let $C_M=P_M|_{1^\perp}$ and
$C=P|_{\mathcal H_{\mathrm{pred}}\cap1^\perp}$. Both are contractions,
and $C_Mf\to Cf$. For a normally ordered CAR word, its image is the
product of creation and annihilation operators with these propagated
modes. The identity $\|a^\dagger(f)\|=\|a(f)\|=\|f\|$ and
factor-by-factor telescoping show convergence in operator norm on every
such fixed word. Their finite span is norm dense in the CAR algebra.
The maps are unital completely positive contractions, so approximation by
these words extends convergence to every fixed CAR observable.
Iterating the contraction estimate proves convergence of each fixed
nested regression expression. Thus both the finite transition matrices
and their fermionic representation approximate the same recorded theory.
The partition modes are used for transition matrices; no finite gradient
energy is attributed to their discontinuous cell indicators.
:::


:::{div} feynman-prose
For the original selected readout, the first memory term remains explicit:
{ref}`(SM.M3) <eq-fg-sm-m3>` compares the exact two-step prediction with
two applications of the channel's one-step conditional average. The
additional term keeps the dependence that leaves the channel subspace and
returns on the next step. Longer excursions produce the memory sum.

The prediction completion retains enough observable functions to set
$\mathsf C=0$ on the enlarged space. Its finite partitions then approximate
that closed evolution with stationary transition matrices. Thus the same
algorithm supplies both descriptions: exact memory for the selected readout,
and an exact Markov extension with convergent finite approximations. The
exact constructions retain the original correlations, and the finite
matrices converge to them at each fixed observation sequence.
:::

:::{prf:theorem} Direct use of reconstruction, concentration, and transfer bounds
:label: thm-sm-direct-existing-machinery

The direct observable representation has the following consequences of the
existing Fractal Gas results.

1. Under the record-completeness conditions of
   {prf:ref}`thm-fractal-set-lossless`, velocity, force, and fitness inputs
   are recovered by {prf:ref}`thm-fractal-set-trajectory`,
   {prf:ref}`thm-fractal-set-force`, and
   {prf:ref}`thm-fractal-set-landscape`. Recorded companion indices, masks,
   and the fixed reconstruction parameters then determine every direct
   channel at the stated sample times.
2. Let an actual continuous joint law $\pi_N$ satisfy
   {prf:ref}`cor-n-uniform-lsi`. For a descriptor map $\mathscr D$ on
   that same state space, define the induced energy on real functions by

   $$
   \mathcal E_{\mathrm{dir}}(f)
   =\int\sum_i\bigl(|\nabla_{x_i}(f\circ\mathscr D)|^2
                    +|\nabla_{v_i}(f\circ\mathscr D)|^2\bigr)d\pi_N,
   $$

   on the domain where $f\circ\mathscr D$ belongs to the weighted Sobolev
   domain of that LSI. Then its law satisfies the exact inequalities

   $$
   \operatorname{Ent}_{\mathscr D_*\pi_N}(f^2)
   \le2C_*\mathcal E_{\mathrm{dir}}(f),\qquad
   \operatorname{Var}_{\mathscr D_*\pi_N}(f)
   \le C_*\mathcal E_{\mathrm{dir}}(f).
   $$

   Thus the previously proved constant is preserved with its induced
   energy. The moment bound of {prf:ref}`lem-ym-lsi-moments` uses the
   Lipschitz constant of the actual pulled-back observable. A bounded
   whole-swarm observable has its own gradient bound; a $1/N$ improvement
   uses the gradient scaling of its actual averaging map.
3. Under the stationary temporal contraction hypothesis of
   {prf:ref}`thm-cluster-decomposition`, direct observables of the same
   Markov state obey its covariance bound. More generally a centered
   past-block observable $F$ ending at time zero and a centered future-block
   observable $G$ beginning at time $t\ge0$ satisfy

   $$
   |\mathbb E[\overline F G]|
   \le M e^{-\lambda t}\sqrt{\mathbb E|F|^2\mathbb E|G|^2}.
   $$

   Thus a force/velocity alignment spanning a transition is covered using
   the separation between its recorded blocks. Under
   {prf:ref}`lem-transfer-matrix-fg`, a bounded direct future functional
   obeys its reflection factorization when its reconstruction commutes
   with the specified time reflection. The same measure and time
   coordinate enter both sides of that identity.
4. For any supplied smooth scalar reconstruction of a direct channel, the
   spatial limits of {prf:ref}`thm-laplacian-convergence` and
   {prf:ref}`lem-lqft-energy-sampling`, or the spacetime limit of
   {prf:ref}`cor-continuum-consistency-conditional`, apply under their
   stated geometric, derivative, sampling, and normalized reconstruction
   error bounds. Their conclusions and rates are unchanged by evaluating
   the identical observable in its invariant coordinates.

The induced energy retains the original full-particle gradient. It also
retains the domain distinction for discrete status variables in
{prf:ref}`prop-kl-status-entropy`. Reconstruction by force normalization or
hard masks is evaluated on this domain before its energy is used; bounded
channel values alone do not assert Sobolev regularity. This formulation
transports the established inequality without assuming an independent
Euclidean-gradient LSI for the descriptor coordinates.
Additional random companion variables retain their conditional distribution
and entropy term; a static state-space LSI is not applied as a path-space
LSI. The block calculation in item 3 uses the actual path law instead.
:::

:::{prf:proof}
**1. Reconstruction commutes with evaluation.** Denote the proved
record-extraction map by $R$. On its reconstruction targets,
$R(\mathcal F)$ gives the recorded $v,F^{\mathrm{visc}},F_i$ and the
other supplied attributes. For a valid sample its color entry is therefore

$$
c_i^a(R(\mathcal F))
=\frac{R(F_i^{\mathrm{visc},a})}
       {\sqrt{\sum_bR(F_i^{\mathrm{visc},b})^2}}
  \exp\!\left(\frac{im\ell_0R(v_i^a)}{\hbar_{\mathrm{eff}}}\right).
$$

The extraction identities make this equal to the entry computed from the
original recorded arrays. Multiplying, conjugating, and summing gives
$q_{ij}(R(\mathcal F))=q_{ij}$, $b_{ijk}(R(\mathcal F))=b_{ijk}$,
and $\Pi_{ijk}(R(\mathcal F))=\Pi_{ijk}$. The same substitution for
fitnesses, companion indices, and distances gives $a_i$ and $s_i^\pm$.
Matching masks and weights give identical numerators and denominators
in $\mathcal A_t$, including its zero-denominator branch. This proves
pointwise equality of the direct observable maps before taking any law
or limit.

**2. Entropy and energy.** Let $\nu=\mathscr D_*\pi_N$ and
$A=\int f^2d\nu=\int(f\circ\mathscr D)^2d\pi_N$. For $A>0$,

$$
\begin{aligned}
\operatorname{Ent}_{\nu}(f^2)
&=\int f(y)^2\log\frac{f(y)^2}{A}\,d\nu(y)\\
&=\int f(\mathscr D(s))^2
       \log\frac{f(\mathscr D(s))^2}{A}\,d\pi_N(s)\\
&=\operatorname{Ent}_{\pi_N}((f\circ\mathscr D)^2)
\le2C_*\mathcal E_{\mathrm{dir}}(f).
\end{aligned}
$$

The case $A=0$ is the zero function in the respective $L^2$ spaces and
has zero entropy.
The induced energy can be evaluated in the original coordinates. Write
$s=(x_1,v_1,\ldots,x_N,v_N)$ and use real coordinates for the descriptor
(real and imaginary parts of complex entries). On a differentiable chart
let $J_{a\ell}(s)=\partial_{s_\ell}\mathscr D_a(s)$. The chain rule gives

$$
\begin{aligned}
\partial_{s_\ell}(f\circ\mathscr D)
 &=\sum_a(\partial_a f)(\mathscr D(s))J_{a\ell}(s),\\
\mathcal E_{\mathrm{dir}}(f)
 &=\int\sum_{a,b}(\partial_a f)(\mathscr D(s))
       (J(s)J(s)^{\mathsf T})_{ab}
       (\partial_b f)(\mathscr D(s))\,d\pi_N(s).
\end{aligned}
$$

For a valid force sample write $u=F_i^{\mathrm{visc}}$, $r=\|u\|>0$,
and $\kappa=m\ell_0/\hbar_{\mathrm{eff}}$. In any differentiable
direction $\delta$ the normalization and phase derivatives are

$$
\delta r=\frac{\sum_bu_b\delta u_b}{r},\qquad
\delta c_i^a=e^{i\kappa v_i^a}
\left(\frac{\delta u_a}{r}
 -\frac{u_a\sum_bu_b\delta u_b}{r^3}
 +i\kappa\frac{u_a}{r}\delta v_i^a\right).
$$

Hence the contraction derivatives needed in $J$ are explicitly

$$
\begin{aligned}
\delta q_{ij}
 &=(\delta c_i)^\dagger c_j+c_i^\dagger\delta c_j,\\
\delta b_{ijk}
 &=\det[\delta c_i,c_j,c_k]+\det[c_i,\delta c_j,c_k]
   +\det[c_i,c_j,\delta c_k],\\
\delta\Pi_{ijk}
 &=(\delta q_{ij})q_{jk}q_{ki}
   +q_{ij}(\delta q_{jk})q_{ki}
   +q_{ij}q_{jk}(\delta q_{ki}).
\end{aligned}
$$

These chart calculations supply the energy integrand wherever the
underlying reconstructed fields are differentiable. Across mask or
companion-selection boundaries, the asserted LSI continues to use the
weak-gradient domain stated in item 2; the chart calculation by itself
does not establish that a discontinuous readout lies in that domain.
 For a bounded real domain function $g$, put $m=\nu g$,
$b=\nu(g^2)$, and $f_\epsilon=1+\epsilon g$, with
$|\epsilon|\|g\|_\infty<1/2$. The expansions below have uniformly
bounded remainders after division by $|\epsilon|^3$:

$$
\begin{aligned}
A_\epsilon&=\nu(f_\epsilon^2)=1+2\epsilon m+\epsilon^2b,\\
\log f_\epsilon^2&=2\epsilon g-\epsilon^2g^2+O(\epsilon^3),\\
f_\epsilon^2\log f_\epsilon^2
 &=2\epsilon g+3\epsilon^2g^2+O(\epsilon^3),\\
\log A_\epsilon&=2\epsilon m+\epsilon^2(b-2m^2)+O(\epsilon^3),\\
A_\epsilon\log A_\epsilon
 &=2\epsilon m+\epsilon^2(b+2m^2)+O(\epsilon^3).
\end{aligned}
$$

Subtracting the last line from the integral of the third gives
$\operatorname{Ent}_\nu(f_\epsilon^2)
=2\epsilon^2(b-m^2)+O(\epsilon^3)$.
The weak-gradient identity
$\nabla(f_\epsilon\circ\mathscr D)
=\epsilon\nabla(g\circ\mathscr D)$ gives
$\mathcal E_{\mathrm{dir}}(f_\epsilon)
=\epsilon^2\mathcal E_{\mathrm{dir}}(g)$.
Dividing the LSI by $2\epsilon^2$ and taking $\epsilon\to0$ proves
$\operatorname{Var}_\nu g\le C_*\mathcal E_{\mathrm{dir}}(g)$.
For an unbounded real domain function take
$g_K=\max(-K,\min(g,K))$. The Sobolev truncation rule gives
$\mathcal E_{\mathrm{dir}}(g_K)\le\mathcal E_{\mathrm{dir}}(g)$,
and $g_K\to g$ in $L^2(\nu)$. Thus the variances converge and the
same inequality holds on the stated domain.
Applying the existing moment theorem to $g\circ\mathscr D$ gives

$$
\log\mathbb E_\nu e^{t(g-\nu g)}
=\log\mathbb E_{\pi_N}e^{t(g\circ\mathscr D-\pi_N(g\circ\mathscr D))}
\le\frac{C_*L^2t^2}{2},
$$

where $L$ is precisely the pulled-back Lipschitz constant used in
{prf:ref}`lem-ym-lsi-moments`.

**3. Temporal blocks and reflection.** Put
$f(s)=\mathbb E[F\mid S_0=s]$ and
$g(s)=\mathbb E[G\mid S_t=s]$. The Markov property separates the past
and future conditional on the intervening state. Applying it at zero
and then at $t$ gives

$$
\mathbb E[\overline F G]
=\mathbb E[\overline{f(S_0)}\,(P_tg)(S_0)]
=\langle f,P_tg\rangle_{\pi}.
$$

Both functions are centered. Conditional Jensen gives
$\|f\|_2^2\le\mathbb E|F|^2$ and
$\|g\|_2^2\le\mathbb E|G|^2$. Thus the already established estimate
$\|P_tg\|_2\le M e^{-\lambda t}\|g\|_2$ yields the claimed bound
by Cauchy--Schwarz. For a direct future functional $F$ whose reconstruction
intertwines the specified reflection, its pullback $\widetilde F$ obeys
$\widetilde{\Theta F}=\Theta\widetilde F$. Under the law of
{prf:ref}`lem-transfer-matrix-fg`, the exact reflected form is consequently

$$
\mathbb E[\overline{\Theta\widetilde F}\,\widetilde F]
=\int\left|\mathbb E[\widetilde F\mid S_0=s]\right|^2d\pi(s)
\ge0.
$$

This uses the factorization already proved for that law, with no
replacement of its generator or its time reflection.

**4. Equality of estimators and convergence bounds.** If two coordinate
representations yield identical reconstructed values $\phi_i$ and identical
weights, then each finite kernel sum agrees term by term:

$$
\sum_jw_{ij}(\phi_j-\phi_i)
=\sum_jw_{ij}(\widetilde\phi_j-\widetilde\phi_i).
$$

Their empirical quadratic energies also agree term by term. The proof of
item 1 and the invariant-coordinate inverse give this equality for the
represented channels. Hence their squared errors against the same
continuum target are equal random variables. The bias, covariance,
bandwidth, and quadrature bounds in the cited spatial and spacetime
theorems therefore transfer with their original constants and applicability
conditions. This step transports those proved estimates; it leaves the
particle transition law fixed. $\square$
:::

:::{prf:proposition} Channel derivatives and the applicable statistical bounds
:label: prop-sm-channel-estimate-routes

Use the actual law, gradient form, and evaluation-stage conventions of
{prf:ref}`thm-sm-direct-existing-machinery`. On a differentiable valid-color
chart, write $u_i=F_i^{\mathrm{visc}}$, $r_i=\|u_i\|$, and
$\kappa=m\ell_0/\hbar_{\mathrm{eff}}$. For a real coordinate variation,

$$
\|\delta c_i\|
\le\frac{\|\delta u_i\|}{r_i}+|\kappa|\|\delta v_i\|.
$$

For valid unit colors this implies

$$
\begin{aligned}
|\delta q_{ij}|&\le\|\delta c_i\|+\|\delta c_j\|,\\
|\delta b_{ijk}|&\le\|\delta c_i\|+\|\delta c_j\|+\|\delta c_k\|,\\
|\delta\Pi_{ijk}|&\le2(\|\delta c_i\|+\|\delta c_j\|+\|\delta c_k\|).
\end{aligned}
$$

For fixed companion indices, the direct doublet amplitude has derivative

$$
\delta a_i=a_i\left[-\frac{\delta(D_i^2)}{4\ell_c^2}
+i\,\delta\theta_i\right],\qquad
\delta\theta_i=
\frac{\delta F_{k(i)}-\delta F_i}{h_S A_i}
-\frac{(F_{k(i)}-F_i)\operatorname{sgn}(F_i)\delta F_i}{h_S A_i^2},
\quad A_i=|F_i|+\varepsilon_{\mathrm{clone}},
$$

where the displayed derivative is used away from $F_i=0$. For the canonical
positive fitness, $F_i>0$ and $\operatorname{sgn}(F_i)=1$. The full fitness
bounds for these derivatives are {prf:ref}`thm-c3-regularity` and
{prf:ref}`thm-unified-cinf-regularity-both-mechanisms`; the force derivative is
that of the actual viscous kernel and recorded force evaluation.
For $z=d/\|d\|$ on a nonzero doublet, the real derivative norm is bounded by
$\|\delta d\|/\|d\|$.

These calculations select the following existing estimate routes.

| Recorded observable | Applicable proved estimate | Quantity to evaluate |
|---|---|---|
| A Sobolev function of the continuous state, including smooth color or doublet contractions | {prf:ref}`cor-quantitative-lsi-final` and {prf:ref}`thm-sm-direct-existing-machinery` | The integrated squared full-state derivative of its actual pullback |
| A real globally Lipschitz pullback | {prf:ref}`lem-ym-lsi-moments` | Its full-state Lipschitz constant, including all normalization factors |
| An average $N^{-1}\sum_i g(Z_i)$ of a fixed bounded single-coordinate test | {prf:ref}`thm-mixing-variance-corrected` | The total entropy relative to the stated product reference and the bound on $g$ |
| A bounded masked whole-swarm channel in a stationary record, including a fixed-length transition block | {prf:ref}`thm-cluster-decomposition` and the temporal-block proof above | Its variance and the separation between the recorded blocks |
| A smooth spatial or spacetime reconstruction | {prf:ref}`thm-sm-laplacian-convergence` or {prf:ref}`cor-cst-inherited-lsi-consistency` | The specified sampling law, kernel bandwidth, derivative constants, and normalized reconstruction error |

For a real bounded state channel $|O|\le B$ and a stationary semigroup with
the constants $M,\lambda$ in {prf:ref}`thm-cluster-decomposition`, sampling
$K$ times at spacing $h>0$ gives the explicit bound

$$
\operatorname{Var}\left(\frac1K\sum_{a=0}^{K-1}O(S_{ah})\right)
\le\frac{B^2}{K}\left(1+
\frac{2M e^{-\lambda h}}{1-e^{-\lambda h}}\right).
$$

The unit-color real and imaginary pair and determinant channels have $B=1$;
$1-\operatorname{Re}\Pi$ has $B=2$. Positive weighted averages with the
zero-denominator convention preserve these bounds. Standard doublet sums
and differences satisfy $|a_i\pm a_{k(i)}|\le2$. Unnormalized displacement
weights or exponential score weights retain their actual moment bounds.

*Proof.* Put $n_i=u_i/r_i$. Differentiation in real coordinates gives
$\delta n_i=(I-n_in_i^{\mathsf T})\delta u_i/r_i$. The matrix in parentheses
is an orthogonal projector. The diagonal phase factors are unitary, and
$\|\operatorname{diag}(n_i)\delta v_i\|\le\|\delta v_i\|$, proving the
color bound. Differentiate the pair and determinant contractions and use
Cauchy--Schwarz and the determinant bound by the product of column norms.
Differentiate the three overlap factors in $\Pi$; each color occurs in two
factors, giving the coefficient two.

For $a_i$, differentiate its logarithmic expression with all reference scales
fixed. The quotient rule gives the displayed $\delta\theta_i$. Normalization
of a doublet is the same orthogonal-projection derivative on $\mathbb R^4$.
For a differentiable weighted average $A=(\sum_Iw_IO_I)/W$, $W=\sum_Iw_I>0$,

$$
\delta A=\frac1W\sum_I\left[w_I\delta O_I+(O_I-A)\delta w_I\right].
$$

This identity accounts for derivatives of the weights and of the denominator.
It applies within a fixed mask chart; membership in the global Sobolev domain
is the domain condition of the established LSI. A jump across a hard threshold
is not removed by a chart derivative bound. The bounded-channel temporal route
uses no derivatives of that mask.

For the temporal variance, stationarity and the covariance estimate give

$$
\begin{aligned}
\operatorname{Var}\left(K^{-1}\sum_aO(S_{ah})\right)
&\le\frac{\operatorname{Var}_\pi O}{K^2}
 \left[K+2M\sum_{r=1}^{K-1}(K-r)e^{-\lambda rh}\right]\\
&\le\frac{B^2}{K}
 \left[1+2M\sum_{r=1}^\infty e^{-\lambda rh}\right].
\end{aligned}
$$

Sum the geometric series. For a channel occupying a block of duration $b$,
replace the lag-$r$ covariance factor by
$\min\{1,M e^{-\lambda(rh-b)}\}$ when $rh\ge b$, and by one for overlapping
blocks; this is the proved block-separation estimate and Cauchy--Schwarz.
Complex channels use the same argument with conjugated covariances, or apply
the real estimate to their two components. No $N^{-1}$ factor is asserted for
an arbitrary whole-swarm readout. $\square$
:::

### The implemented transition in reconstructed coordinates

:::{div} feynman-prose
Imagine stopping the simulation just before an update. To restart it, you
need everything the next step will read: walker states, the scheduling phase,
and any retained geometry or auxiliary tensors. Given that state and the fresh
random inputs, the code determines the next state. Encoding this complete
state therefore gives a direct recipe for the next encoded state too.

An observable can need more information than the next update does. A force
alignment measured across a step may use intermediate arrays and a validity
mask. Keep those in the transition record, and the same recipe carries their
joint history law. The theorem below implements this construction for every
finite recorded history.

There is also an equilibrium evolution built from the energy form on its
identified LSI law. Its time $\sigma$ measures that derived evolution. The
table keeps it alongside the algorithm's update times so we can apply each
estimate to the law and clock for which it was proved.
:::

:::{prf:definition} Complete update state and the two field evolutions
:label: def-sm-complete-update-law

Let $s$ contain the walker state and every retained variable read by the next
update: the scheduling phase, retained geometry and auxiliary tensors when
used, and the fixed run parameters. Time-dependent external inputs are indexed
explicitly. Write $\xi$ for the fresh random inputs of one update and
$m(d\xi)$ for their joint law. Companion draws can be realized by inverse
cumulative probabilities applied to uniform variables; their state dependence
then belongs to the update map $T_h(s,\xi)$.

For a conservative step and a killed step, respectively, set

$$
P_hf(s)=\int f(T_h(s,\xi))m(d\xi),\qquad
Q_hf(s)=\int\chi(s,\xi)f(T_h(s,\xi))m(d\xi),
$$

where $\chi$ is the survival indicator and the state after killing is excluded
from $Q_h$. A retained random output is part of the corresponding recorded
transition, even when it is unnecessary for the next update. This distinguishes
the Markov state from a complete transition record.

The notation in the following constructions is fixed by this table.

| Object | Law or operator | Source of its identification |
|---|---|---|
| Finite recorded history | actual initial law and ordered $P_h$ or $Q_h$ kernels | implemented update and record coverage |
| Conservative stationary evolution | $\pi_NP_h=\pi_N$ | the conservative convergence result in its established regime |
| Quasi-stationary evolution | $\nu_NQ_h=\alpha_h\nu_N$ | the killed-chain QSD result |
| Stationary Doob evolution | $P_h^\eta=\alpha_h^{-1}\eta^{-1}Q_h\eta$, $\pi_N^\eta=\eta\nu_N$ | {prf:ref}`prop-kl-doob-transform`, with $\nu_N\eta=1$ |
| Equilibrium energy evolution | $T_\sigma^{\mathrm{eq}}=e^{-\sigma H_{\mathrm{eq}}}$ on the law of its established LSI | {prf:ref}`thm-ym-equilibrium-form-construction` |

The last time coordinate is denoted by $\sigma$. The algorithmic observation
time remains $h$, $mh$, or $t$ in the already specified continuous model.
:::

:::{prf:theorem} Exact transition and history isomorphism for the recorded algorithm
:label: thm-sm-instantiated-record-transition

Use the complete records of {prf:ref}`def-fractal-set-record-coverage` and
their inverse maps $E=\operatorname{Enc}$, $D=\operatorname{Dec}$. At a
Markov boundary, encode the complete state and its required header; for a
transition observable, retain the complete transition record. On the encoded
image the actual conservative transition is

(eq-fg-sm-k1)=
$$
\widehat P_hg(c)
=\int g\bigl(E T_h(Dc,\xi)\bigr)m(d\xi).
\tag{SM.K1}
$$

The killed formula contains the additional factor $\chi(Dc,\xi)$. With
$Uf=f\circ D$ and $\widehat\pi=E_\#\pi$, these kernels satisfy

(eq-fg-sm-k2)=
$$
\widehat P_hU=UP_h,\qquad
\widehat P_h^*=UP_h^*U^{-1}
\quad\text{on }L^2(\widehat\pi)
\tag{SM.K2}
$$

whenever $\pi$ is the identified invariant law. For the established
strongly continuous realization the generator is

(eq-fg-sm-k3)=
$$
\widehat L=ULU^{-1},\qquad
\operatorname{Dom}\widehat L=U\operatorname{Dom}L.
\tag{SM.K3}
$$

Every finite integrable history observable, including the direct channels
and their recorded masks, has exactly the same expectation after encoding.
For bounded state observables and $0=t_0<t_1<\cdots<t_n$ this identity reads

(eq-fg-sm-k4)=
$$
\mathbb E_\pi\prod_{j=0}^n f_j(S_{t_j})
=\left\langle1,M_{f_0}P_{t_1}M_{f_1}
 P_{t_2-t_1}\cdots P_{t_n-t_{n-1}}M_{f_n}1\right\rangle_\pi.
\tag{SM.K4}
$$

The encoded expression replaces each factor by its unitary transport.
These identities identify the field evolution specified by the update
itself, with no comparison to an independently chosen differential operator.
:::

:::{prf:proof}
**One step.** Substitute $g=Uf$ in {ref}`(SM.K1) <eq-fg-sm-k1>`. The reconstruction identity
$DEs'=s'$ on every covered output gives

$$
\widehat P_hUf(c)
=\int f\bigl(DE T_h(Dc,\xi)\bigr)m(d\xi)
=P_hf(Dc)=UP_hf(c).
$$

The killed identity has the same integrand multiplied by its unchanged
survival indicator. Fresh random inputs can instead be integrated through
their conditional kernels; inverse-cumulative sampling shows that this is
the same integral. The completed intermediate state retains any pre-cloning
data needed in a subsequent substep. Thus the implemented order, cloning
followed by kinetics, gives the backward-operator order $P_h=C_hK_h$.

**Hilbert space and domain.** The unitary is
{prf:ref}`prop-fractal-set-analytic-transfer`. For $f,g\in L^2(\pi)$,

$$
\langle Uf,\widehat P_hUg\rangle_{\widehat\pi}
=\langle f,P_hg\rangle_\pi
=\langle UP_h^*f,Ug\rangle_{\widehat\pi},
$$

which proves the adjoint identity. In continuous time,

$$
\frac{\widehat P_tUf-Uf}{t}
=U\frac{P_tf-f}{t}.
$$

An $L^2$ limit exists on one side exactly when it exists on the other,
because $U$ is an isometry onto. This proves both the operator and its full
domain in {ref}`(SM.K3) <eq-fg-sm-k3>`. An existing core $\mathcal C$ for $L$ is consequently
carried to a core $U\mathcal C$; no derivative of a Borel decoding section
is taken.

**Histories.** Condition first on $S_{t_{n-1}}$ in the last observable in
{ref}`(SM.K4) <eq-fg-sm-k4>`, then repeat toward $t_0$. This gives the displayed operator product.
The identity $UM_fU^{-1}=M_{Uf}$ and {ref}`(SM.K2) <eq-fg-sm-k2>` cancel every intervening
$U^{-1}U$. For a recorded block $r_j=R(s_{j-1},\xi_j)$ replace its step
kernel by the signed or complex kernel

$$
K_h^{A_j}f(s)
=\int A_j(R(s,\xi))f(T_h(s,\xi))m(d\xi).
$$

Applying the same decoder substitution to each $K_h^{A_j}$ proves the
identity for block observables. General integrable history functions follow
by equality of the pushforward history measures, first on cylinder sets and
then on their generated sigma algebra. Independence of walkers is never
used.

**Conditioning.** For a QSD and a history ending at $mh$, its conditional
expectation is the corresponding product of killed kernels divided by
$\nu_NQ_h^m1=\alpha_h^m$. For the Doob process, multiplication of its
one-step factors telescopes:

(eq-fg-sm-k5)=
$$
\pi_N^\eta(ds_0)\prod_{j=1}^mP_h^\eta(s_{j-1},ds_j)
=\alpha_h^{-m}\eta(s_m)\nu_N(ds_0)
 \prod_{j=1}^mQ_h(s_{j-1},ds_j).
\tag{SM.K5}
$$

The final factor $\eta(s_m)$ distinguishes this stationary history law
from the QSD law conditioned on survival to $mh$. Encoding preserves this
factor as well as the kernels. Invariance of $\eta\nu_N$ follows directly
by integrating $\alpha_h^{-1}\eta^{-1}Q_h(\eta f)$ against it.
:::

:::{div} feynman-prose
The history formula keeps each observation in its proper place between
updates. This is why an intermediate force, a mask, or a companion draw can
be included without treating the walkers or successive observations as
independent. Encoding changes the coordinates of that same calculation.

The survival weights make a useful distinction concrete. Start from the
quasi-stationary law and keep only histories surviving $m$ steps: their
normalizing factor is $\alpha_h^m$. For the stationary Doob process, the
successive ratios of $\eta$ cancel, leaving an additional endpoint weight
$\eta(s_m)$. Thus two equally weighted surviving histories can receive
different Doob weights according to their endpoints. Formula {ref}`(SM.K5) <eq-fg-sm-k5>` carries
that weight through the encoding exactly.

For smooth readouts in the generator domain, we can also express the
evolution through their derivatives. That is the purpose of the next formula;
the integral transition already applies to bounded readouts with hard masks.
:::

:::{prf:proposition} Differential coefficients of the existing direct observables
:label: prop-sm-direct-update-coefficients

For the continuous realization in {prf:ref}`def-kl-full-generator`, let
$y^\alpha=\mathscr D^\alpha(s)$ be direct descriptor coordinates on a
chart where the actual pullback belongs to its generator domain and is
twice differentiable. For a smooth scalar $f$ of these coordinates,

(eq-fg-sm-k6)=
$$
\begin{aligned}
L(f\circ\mathscr D)(s)
&=\sum_\alpha\beta^\alpha(s)\partial_\alpha f(\mathscr D(s))
 +\sum_{\alpha,\beta}A^{\alpha\beta}(s)
       \partial_{\alpha\beta}f(\mathscr D(s))\\
&\quad+\int[f(\mathscr D(s'))-f(\mathscr D(s))]r_N(s,ds'),\\
\beta^\alpha
&=b_N\cdot\nabla\mathscr D^\alpha
  +\operatorname{tr}(a_N\nabla^2\mathscr D^\alpha),\qquad
A^{\alpha\beta}
=\nabla\mathscr D^\alpha{}^{\mathsf T}a_N\nabla\mathscr D^\beta.
\end{aligned}
\tag{SM.K6}
$$

The coefficients, jump images, and domains descend to a closed descriptor
evolution precisely through the criterion in
{prf:ref}`prop-sm-direct-markov-intertwining`. Formula {ref}`(SM.K1) <eq-fg-sm-k1>` applies
directly to bounded masked channels without a differentiability assertion.
:::

:::{prf:proof}
The first derivative is
$\nabla(f\circ\mathscr D)=\sum_\alpha f_\alpha\nabla\mathscr D^\alpha$.
The second derivative is

$$
\nabla^2(f\circ\mathscr D)
=\sum_\alpha f_\alpha\nabla^2\mathscr D^\alpha
 +\sum_{\alpha,\beta}f_{\alpha\beta}
   \nabla\mathscr D^\alpha\nabla\mathscr D^\beta{}^{\mathsf T}.
$$

Insert both into the actual diffusion-jump generator. This produces
{ref}`(SM.K6) <eq-fg-sm-k6>`, including the second-derivative drift and the entire jump
displacement. The force-normalization differentials and probability
derivatives are those already computed in
{prf:ref}`prop-sm-channel-estimate-routes`. On a chart they enter these
two derivatives; at a hard mask boundary the exact integral kernel remains
the definition. Constancy of the transition probabilities on descriptor
fibers is exactly the previously proved quotient criterion. Retaining the
complete encoded state gives {ref}`(SM.K1) <eq-fg-sm-k1>` without requiring that constancy.
:::

:::{div} feynman-prose
Now follow a collision through the common routine's assignments. A group
update uses its walkers' velocities and their common mean. If the companion
interface permits overlapping groups, a walker can be written twice. Each
mean uses the original velocities, and the last visited group supplies that
walker's final value. The three-walker calculation below shows why that
general update must retain the group order.

The current `make physics` application supplies mutual disjoint pairs to
this routine. Its sampling law excludes the overlapping example. After the
general calculation, {prf:ref}`cor-sm-physics-paired-cloning` specializes the
update to those pairs and proves its momentum conservation and permutation
equivariance.
:::

:::{prf:proposition} Selected collision increments in the implemented record
:label: prop-sm-implemented-collision-increments

For the common collision routine in
`src/fragile/fractalai/core/cloning.py` and
`src/fragile/physics/fractal_gas/cloning.py`, let $c_i$ be the sampled
companion and $I_i$ the
accepted-cloning indicator. The acceptance probability for an alive walker is

$$
p_i=\left[\frac{F_{c_i}-F_i}
 {p_{\max}(F_i+\varepsilon_{\mathrm{clone}})}\right]_0^1,
\qquad I_i=1_{\{U_i<p_i\}}.
$$

The status-bearing `fractalai` interface also has its specified forced
revival branch; the current physics application uses an all-alive state.
Conditional
on the state, companions, and accepted indicators, the position increment is

(eq-fg-sm-k7)=
$$
x_i'-x_i=I_i(x_{c_i}-x_i+\sigma_x\zeta_i),\qquad
\mathbb E[x_i'-x_i\mid s,c,I]=I_i(x_{c_i}-x_i).
\tag{SM.K7}
$$

For each active companion $j$, put
$G_j=\{j\}\cup\{i:I_i=1,\ c_i=j,\ i\ne j\}$ and
$\bar v_j=|G_j|^{-1}\sum_{i\in G_j}v_i$, using the pre-cloning velocities.
The implementation visits the active companion indices in increasing order.
If $j_*(i)$ is the last visited group containing $i$, its output is

(eq-fg-sm-k8)=
$$
v_i'=\alpha v_i+(1-\alpha)\bar v_{j_*(i)},
\tag{SM.K8}
$$

with $v_i'=v_i$ for walkers in no group. Within one disjoint group the
relative kinetic energy is multiplied by $\alpha^2$. For overlapping
groups, {ref}`(SM.K8) <eq-fg-sm-k8>` determines the full increment and its covariance.
:::

:::{prf:proof}
The position assignment and centered Gaussian give {ref}`(SM.K7) <eq-fg-sm-k7>`. For each group,
the code computes $u_i=v_i-\bar v_j$ from the original velocity array and
writes $\bar v_j+\alpha u_i$ to the output array. The last write gives
{ref}`(SM.K8) <eq-fg-sm-k8>`. For a single group,
$\sum_{i\in G_j}u_i=0$, so its updated sum is unchanged and
$\sum|u_i'|^2=\alpha^2\sum|u_i|^2$.

For an explicit overlapping event, use indices $0,1,2$,
$v=(0,2,8)$, $c=(1,2,0)$, $I=(1,1,0)$, and $\alpha=1/2$.
The first group writes $(v_0',v_1')=(1/2,3/2)$.
The second writes $(v_1',v_2')=(7/2,13/2)$. Thus the final vector is

(eq-fg-sm-k9)=
$$
(1/2,7/2,13/2),\qquad \sum_i v_i'=21/2,
\quad \sum_i v_i=10.
\tag{SM.K9}
$$

Relabeling $0\leftrightarrow2$ reverses the order of these two group
writes. Undoing the relabeling then gives $(1/2,3/2,13/2)$.
Consequently the recorded collision map is not pathwise permutation
equivariant on this event. For a general companion interface admitting the indicated draws, this
event has positive gate probability for strictly increasing fitnesses.
The mutual-pair law of the current physics application excludes it, as
proved in {prf:ref}`cor-sm-physics-paired-cloning` below.
The symmetry premise of {prf:ref}`thm-qsd-exchangeability` must therefore
be checked for the actual averaged kernel; it cannot be discharged by
claiming that these recorded group writes commute. The exact record and
CAR constructions above do not require exchangeability.

For two walkers in one nontrivial group and $0<\alpha<1$, velocity reversal
preserves the length of the relative velocity. Another clone contracts it
again, to $\alpha^2|u|$, whereas reversing the original jump would require
expansion by $1/\alpha$. Thus the selected jump component has no such
momentum-reversed jump on this event. A positive Doob weight multiplies
existing jump rates and preserves their support. This calculation concerns
the jump component. The complete finite-step transition also includes its
kinetic kernel and is analyzed by {ref}`(SM.K1) <eq-fg-sm-k1>`, with the implemented order.
:::

:::{div} feynman-prose
For `make physics`, picture the walkers arranged in pairs, with one walker
left over when the population is odd. Within each pair, the two fitness
differences have opposite signs, so at most one walker accepts cloning.
That event updates the pair's velocities together: their sum stays fixed,
and their relative velocity is multiplied by $\alpha$. The leftover walker's
self-companion produces zero cloning probability.

Because the pairs are disjoint, their assignments cannot overwrite one
another. We can therefore sum their conservation identities over the swarm.
Uniform random pairing also treats relabeled walkers alike, giving the
cloning symmetry proved below when the fitness inputs are relabeled with
them.
:::

:::{prf:corollary} Mutual-pair cloning in the current physics application
:label: cor-sm-physics-paired-cloning

The `make physics` application constructs its gas from
`src/fragile/physics/fractal_gas/euclidean_gas.py`. Its companion sampler
`random_pairing_fisher_yates` returns a uniformly random mutual pairing,
with one self-companion when $N$ is odd. All walkers in this implementation
are alive. For this sampling law the accepted collision groups are disjoint,
and the complete cloning step satisfies

(eq-fg-sm-k10)=
$$
\sum_i v_i'=\sum_i v_i,
\qquad
\sum_i|v_i'|^2
=\sum_i|v_i|^2-(1-\alpha^2)
 \sum_{\{i,j\}\ {\rm accepted}}\frac{|v_i-v_j|^2}{2}.
\tag{SM.K10}
$$

Its cloning kernel is permutation equivariant when the input fitness vector
is relabeled with the walker state. The overlapping event in {ref}`(SM.K9) <eq-fg-sm-k9>`
belongs to the more general companion interface; it has probability zero
under this particular mutual-pair law.
:::

:::{prf:proof}
For $N=2m$, each fixed unordered matching is represented by $2^m m!$
permutations: choose the order of its pairs and the order inside each pair.
The shuffle is uniform over $(2m)!$ permutations. For $N=2m+1$, the
unpaired walker is last and each matching with its specified singleton
again has $2^m m!$ representatives among $(2m+1)!$ permutations. These
counts are unchanged by relabeling. In particular $c_{c_i}=i$.

For a mutual pair with unequal nonnegative fitnesses, the two score
numerators are opposite. Their positive denominators preserve those signs,
so at most the lower-fitness walker has a positive acceptance probability.
Equal fitnesses give zero acceptance on both sides. A self-companion has
score zero. Consequently an accepted event updates exactly the two members
of one pair, and different events have disjoint members. In each pair the
relative vectors are $(v_i-v_j)/2$ and its negative. Their squared norm
sum is $|v_i-v_j|^2/2$. Apply the single-group calculation in
{prf:ref}`prop-sm-implemented-collision-increments` and sum over the
disjoint pairs to obtain {ref}`(SM.K10) <eq-fg-sm-k10>`.

Relabel a realized mutual matching, its fitness vector, gate uniforms,
and Gaussian position jitters together. Each two-member velocity update
and each selected position assignment then gives the relabeled original
output. The independent uniforms and Gaussian jitters have the same law,
as does the uniform matching. This proves equivariance of the cloning
kernel for these inputs. It identifies the cloning factor used by the
physics application; the complete kernel also retains its geometry,
fitness evaluation, scheduling phase, and kinetic step in {ref}`(SM.K1) <eq-fg-sm-k1>`.
:::


:::{div} feynman-prose
For one doublet, keeping both the sum and the difference lets you recover
its two amplitudes. Averaging over the swarm discards that information.
Every mutual pair contributes a difference in each orientation: one is the
negative of the other. With equal weights at the two ends, they cancel
exactly, while the sum channel counts the component average twice. This
argument works for the score-directed amplitudes too. A role mask can select
the two ends differently; the weighted formula below retains that imbalance.

The cancellation also tells us how to construct the simulator's mode space.
A zero channel and a duplicate channel supply no additional independent
modes. Their linear relations give null directions in the centered Gram
matrix. Quotient those directions before normalizing a basis and constructing
its exterior and CAR operators. Invertibility of the unaveraged doublet
readout does not restore information removed by the frame average.
:::

:::{prf:corollary} Exact cancellation and redundancy of the paired doublet averages
:label: cor-sm-paired-doublet-cancellation

For the mutual-pair companion map $c_{c_i}=i$ of
{prf:ref}`cor-sm-physics-paired-cloning`, use the recorded amplitudes
$a_i$ and readouts $s_i^\pm=a_i\pm a_{c_i}$ of
{prf:ref}`thm-sm-direct-doublet-readout-isomorphism`.
For any nonnegative effective weights $w_i$ with
$W=\sum_iw_i>0$,

(eq-fg-sm-m5)=
$$
\frac1W\sum_iw_i s_i^-
 =\frac1W\sum_i(w_i-w_{c_i})a_i.
\tag{SM.M5}
$$

Consequently pair-symmetric weights give a zero difference average
and a sum average twice the component average:

(eq-fg-sm-m6)=
$$
\frac1W\sum_iw_i s_i^-=0,\qquad
\frac1W\sum_iw_i s_i^+
=\frac2W\sum_iw_i a_i
\quad(w_i=w_{c_i}).
\tag{SM.M6}
$$

In the current all-alive mutual-pair application, the unsplit valid
frame averages in `_compute_su2_operators` have precisely these
symmetric weights. Thus `su2_doublet_diff` is identically zero and
`su2_doublet` is twice `su2_component` in exact arithmetic, in both
standard and score-directed modes. Their directed variants have
the same identities. Walker-role masks can break pair symmetry;
their difference channels are given by
{ref}`(SM.M5) <eq-fg-sm-m5>`.

All autocorrelations of the unsplit difference series vanish, and
the sum-series autocorrelation is four times the component-series
autocorrelation, including connected subtraction with the same
normalization. The corresponding centered mode space must quotient
these zero and dependent directions before constructing its mass
matrix or a faithful exterior basis.
:::

:::{prf:proof}
An involution is a bijection. Reindexing $j=c_i$ gives
$\sum_iw_i a_{c_i}=\sum_jw_{c_j}a_j$.
Subtracting proves {ref}`(SM.M5) <eq-fg-sm-m5>`, and adding
under $w_i=w_{c_i}$ proves the second identity in
{ref}`(SM.M6) <eq-fg-sm-m6>`. Self-companions obey both formulas
as well.

The implementation forms the two-hop readout by gathering the
already formed amplitude at the companion index. Every companion
index of a complete mutual pairing is valid and every walker is
alive. Its unsplit averaging mask therefore equals one at both
ends of each pair. Replacing each amplitude by its score-directed
version leaves the reindexing calculation unchanged. A role-restricted
average uses the role indicator in $w_i$, which need not agree at
the two ends; the general identity retains that indicator exactly.

The series identities hold before temporal averaging. Multiplying
the sum series at two times gives the factor four, and subtracting
the product of its means gives the same factor. The zero series
has zero covariance at every lag. Centering preserves all these
linear relations, so the $L^2$ Gram matrix has the associated
null directions. This explicitly supplies the zero-norm quotient
required by {prf:ref}`thm-lqft-oriented-word-algebra` for these
implemented channel modes. Floating-point summation may leave
roundoff-sized residuals; it does not remove the exact relation.
:::


:::{prf:remark} Observation schedule and the already proved continuum scaling
:label: rem-sm-actual-step-and-clock

For `clone_every` equal to $q>1$, the homogeneous completed state includes
$\ell\in\mathbb Z/q\mathbb Z$ with $\ell'=\ell+1$. Its observable
$e^{2\pi i\ell/q}$ is an eigenfunction of the full transition with
eigenvalue $e^{2\pi i/q}$. Hence a strict centered mixing estimate on the
entire phase-augmented space would fail. A convergence estimate for observations
at one scheduling phase uses the actual $q$-step kernel on that phase;
intermediate observations retain their ordered phase-dependent kernels.

The fixed-step gate in {ref}`(SM.K7) <eq-fg-sm-k7>` has order-one acceptance probability. The
finite-attempt-rate equation of {prf:ref}`def-cloning-generator` is the
continuous realization specified in
{prf:ref}`rem-mean-field-attempt-scaling`. Its infinitesimal acceptance
scaling is used only in that realization or a proved scaling limit.
All finite-step identities above hold at the implemented timestep.
:::


:::{prf:corollary} Fermionic lift of the direct observable isomorphism
:label: cor-sm-direct-fock-isomorphism

Let $V$ be the unitary pullback of
{prf:ref}`thm-sm-direct-measure-isomorphism`, restricted to centered
functions. Its codomain is the centered subspace of the represented
$\sigma(q)$-measurable observables. Then

$$
\Gamma_-(V)=\bigoplus_{k\ge0}\Lambda^kV
$$

is unitary between their fermionic Fock spaces, maps vacuum to vacuum,
and intertwines the CAR generators:

$$
\Gamma_-(V)a^\dagger(f)\Gamma_-(V)^{-1}=a^\dagger(Vf),
\qquad
\Gamma_-(V)a(f)\Gamma_-(V)^{-1}=a(Vf).
$$

For the induced dynamics of
{prf:ref}`prop-sm-direct-markov-intertwining`, it also intertwines the
Fock transition operators. Consequently all finite operator-word vacuum
matrix elements constructed from these bounded generators and transitions
are identical in the two representations. The record-process Fock space
and its replica realization are those of
{prf:ref}`thm-lqft-record-fock-reconstruction` and
{prf:ref}`thm-lqft-replica-isomorphism`.
:::

:::{prf:proof}
For decomposable wedges the Gram entries obey
$\langle Vf_i,Vg_j\rangle=\langle f_i,g_j\rangle$, so their determinant
inner products agree. The inverse on each sector is $\Lambda^k(V^{-1})$.
Taking the direct sum proves unitarity, and the zeroth sector is the
identity. On a decomposable wedge $\eta$,

$$
\Gamma_-(V)a^\dagger(f)\eta
=\Gamma_-(V)(f\wedge\eta)
=Vf\wedge\Gamma_-(V)\eta
=a^\dagger(Vf)\Gamma_-(V)\eta.
$$

Boundedness extends the identity, and taking adjoints gives the annihilator
identity. If $VP_t^{\rm dir}=P_t^{\rm rec}V$ on the represented subspace,
applying this equality to every wedge factor gives
$\Gamma_-(V)\Gamma_-(P_t^{\rm dir})
=\Gamma_-(P_t^{\rm rec})\Gamma_-(V)$.
Insert these conjugation identities into a finite operator word; adjacent
$\Gamma_-(V)^{-1}\Gamma_-(V)$ factors cancel. The two surviving vacuum
vectors agree, proving equality of the entire matrix element. $\square$
:::

:::{prf:proposition} Generator of the established replica lift
:label: prop-sm-replica-generator

For a strongly continuous contraction semigroup $P_t$ on the centered mode
space $\mathcal H$ of {prf:ref}`thm-lqft-record-fock-reconstruction`, let $L$
be its generator. On wedges with $f_r\in\operatorname{Dom}L$,

$$
L^{(k)}(f_1\wedge\cdots\wedge f_k)
=\sum_{r=1}^k f_1\wedge\cdots\wedge Lf_r\wedge\cdots\wedge f_k,
\qquad L^{(0)}=0.
$$

The same formula holds after the invariant-coordinate unitary in
{prf:ref}`cor-sm-direct-fock-isomorphism`. Each factor $f_r$ is an observable
of a complete swarm state; its $L$ contains the interactions in that original
process. Under {prf:ref}`thm-lqft-replica-isomorphism`, the different factors
are independent replicas before antisymmetrization.

*Proof.* The continuous multilinear wedge map is bounded by the product of
factor norms. Telescope
$P_tf_1\wedge\cdots\wedge P_tf_k-f_1\wedge\cdots\wedge f_k$ by replacing
one factor at a time, divide by $t$, and use
$(P_tf_r-f_r)/t\to Lf_r$ and $P_tf_j\to f_j$. This proves the formula on
the stated domain. The replica unitary identifies the transitions with
$P_t^{\otimes k}$ on its antisymmetric sector, and the same difference
quotients intertwine their generators. $\square$
:::

:::{prf:proposition} Criterion for equality with a specified field evolution
:label: prop-sm-field-generator-comparison

Let $\mathcal W$ be a specified unitary from the represented record Hilbert
space, or its Fock lift, onto a proposed field Hilbert space. Let $K$ be the
record generator and $K_{\mathrm{field}}$ the field generator, both generating
strongly continuous semigroups. If a common dense domain $\mathcal C$ is a
core for $\mathcal W K\mathcal W^{-1}$ and $K_{\mathrm{field}}$, and

$$
K_{\mathrm{field}}f=\mathcal W K\mathcal W^{-1}f
\qquad(f\in\mathcal C),
$$

then their semigroups intertwine, their spectra agree, and corresponding
operator correlations agree when the state and observables are also transported
by $\mathcal W$.

*Proof.* Equality on the common core gives equality of the closed generators,
since each is the closure of its restriction. Uniqueness of the semigroup
with that generator gives
$T_t^{\mathrm{field}}=\mathcal W T_t^{\mathrm{rec}}\mathcal W^{-1}$.
For a resolvent parameter $z$, conjugation gives
$(z-K_{\mathrm{field}})^{-1}=\mathcal W(z-K)^{-1}\mathcal W^{-1}$.
For correlations, conjugate each observable and each transition; adjacent
$\mathcal W^{-1}\mathcal W$ factors cancel, as in the Fock isomorphism proof.
This is the operator comparison in
{prf:ref}`prop-sm-direct-markov-intertwining` applied to the specified field
model. $\square$
:::

:::{prf:example} An interaction not supplied by second quantization alone
:label: ex-sm-replica-interaction-comparison

On $\Lambda^*\mathbb C^2$, put $n_j=a_j^\dagger a_j$ and
$V=\lambda n_1n_2$, with $\lambda\ne0$. This operator is zero on the vacuum
and on the one-mode sector, and equals $\lambda I$ on the two-mode sector.
It cannot equal $d\Gamma(A)$ for a one-mode operator $A$: restriction to the
one-mode sector would force $A=0$, whereas the two-mode restriction is nonzero.
Consequently adding a proposed inter-mode interaction to a lifted field
generator requires the generator comparison above. This does not remove the
interactions already present within each complete-swarm generator $L$.
:::

:::{prf:proposition} Exact exterior meaning of the implemented baryon correlator
:label: prop-sm-baryon-exterior-correlator

For three numerical color vectors in $\mathbb C^3$, with
$E=e_1\wedge e_2\wedge e_3$, the implemented complex determinant is

$$
B_{ijk}=\det[c_i,c_j,c_k]
=\langle E,c_i\wedge c_j\wedge c_k\rangle.
$$

For source and sink color matrices $C_s,C_t\in\mathbb C^{3\times3}$,

$$
\overline{B_s}B_t=\det(C_s^\dagger C_t).
$$

Thus the complex-determinant correlator computed by
`new_channels/baryon_triplet_channels.py` is exactly the masked empirical
average of the real part of this exterior matrix element, with the
configured connected subtraction. It uses the joint law of the recorded
color samples. The operator series using $|B|$ and the other score or flux
modes are their separately specified functions.
:::

:::{prf:proof}
Expand each color vector in the orthonormal basis. Every term with a
repeated basis index vanishes. Sorting the six remaining basis wedges into
$e_1\wedge e_2\wedge e_3$ produces precisely the six permutation signs in
the determinant. This proves the first equality, including its sign under
exchanging two columns. For the two-time identity,

$$
\det(C_s^\dagger C_t)
=\det(C_s^\dagger)\det C_t
=\overline{\det C_s}\det C_t.
$$

The code multiplies the conjugated source determinant by the sink
determinant, takes its real part, and averages over valid source/sink
pairs. The identity holds for each pair before masking, averaging, or
subtracting the configured means, so those operations preserve it.

The distinction from the record-replica determinant is the placement of
expectation: generally
$\mathbb E\det(C_s^\dagger C_t)\ne\det\mathbb E(C_s^\dagger C_t)$.
For a concrete unit-column example let $C_s=I_3$ and
$C_t=\operatorname{diag}(X,X,1)$ with equally likely $X=1,-1$.
Then every determinant is $X^2=1$, whereas
$\mathbb E(C_s^\dagger C_t)=\operatorname{diag}(0,0,1)$ has determinant
zero. The replica theorem specifies exactly the product law under which
its determinant-of-covariances formula holds. $\square$
:::

(sec-sm-matter-sector)=
## 4. Optional Fermionic and Clifford Representations

:::{div} feynman-prose
Start with linearly independent recorded modes, after removing combinations with
zero norm. Each mode is a function of a complete swarm state. To build an
alternating two-mode observable, use two independent copies of the whole
swarm and subtract the expression with the modes exchanged. Each copy keeps
all the interactions among its own walkers.

Inserting another mode into these alternating observables gives the exterior
product. Exchanging two insertions reverses the sign; repeating one gives
zero. The replica inner product also distinguishes all the ordered basis
products, so the representation is faithful: no further relation is hidden
by the construction. These are the derived identities used in
{prf:ref}`axm-sm-grassmann` below.

The stochastic update performs a different operation: it averages future
observables with the transition probabilities. Composing updates advances
the recorded process; composing insertions builds its alternating observable
sectors. The exact encoded transition supplies the evolution of each replica,
and the adjoint of insertion supplies the CAR contraction. The following
corollary also identifies the separate dual symbols used in a Euclidean
Grassmann integral. Clifford and Dirac matrices give further representations;
comparison with a specified gauge or Yukawa evolution uses the stated
generator domains and intertwining maps.
:::

:::{prf:theorem} Exact cloning antisymmetry and its error for raw scores
:label: thm-sm-cloning-antisymmetry

For positive denominators $a_i=V_i+\varepsilon_{\mathrm{clone}}$,

$$
a_i S_i(j)=-a_jS_j(i)=V_j-V_i,
\qquad
S_i(j)+S_j(i)=\frac{(V_j-V_i)^2}{a_i a_j}.
$$

If $a_i,a_j\ge a_*>0$, the last quantity is at most
$(V_j-V_i)^2/a_*^2$. Thus approximate raw antisymmetry requires control of
the denominators as well as the fitness difference.
:::

:::{prf:proof}
Multiply each score by its denominator for the first identity. Putting the
two raw scores over a common denominator gives the second. The denominator
lower bound gives the inequality. This is the direct calculation in
{prf:ref}`thm-cloning-antisymmetry-lqft`. $\square$
:::

:::{prf:theorem} Exclusion of opposing score-based directions
:label: thm-sm-exclusion-principle

At one fixed state, with positive denominators, a strictly positive-score
rule makes precisely one of the opposing directions eligible when
$V_i\ne V_j$, and neither when $V_i=V_j$. This statement concerns eligibility,
not whether a Bernoulli trial accepts it. It imposes no bound on walker
occupancy of a spatial state and does not cover a separate forced-revival rule.
:::

:::{prf:proof}
The score signs are the signs of $V_j-V_i$ and $V_i-V_j$, respectively.
They are opposite unless both are zero. $\square$
:::

:::{div} feynman-prose
The bar on $\bar\psi$ needs care here. In the Euclidean polynomial algebra,
it names a generator from the dual mode space. Multiplying by it adds an
exterior factor, with the same alternating rule as the unbarred generators.
On the Hilbert exterior space, the adjoint of insertion contracts a mode:
it lowers the degree and uses the recorded inner product. This contraction
produces the mixed CAR identity. The two operations therefore have different
products and different jobs, even though both are built from the same
recorded modes. The corollary makes each correspondence explicit.
:::

:::{prf:corollary} Derived exterior representation and Euclidean dual symbols
:label: axm-sm-grassmann

For the finite recorded mode space $E$ in
{prf:ref}`thm-lqft-oriented-word-algebra`, its alternating insertion
operators have the faithful exterior representation
$\mathsf C(e_i)\leftrightarrow\psi_i$.
Anticommutation and nilpotency are the derived identities
{ref}`(LQ.A2) <eq-fg-lq-a2>`, and independence of ordered monomials
is proved by their replica norms. An index $(i,a)$ labels a basis
mode only after the corresponding recorded functions have been
identified and their zero-norm relations removed.

The Euclidean polynomial algebra is $\Lambda(E\oplus E^\vee)$ from
{prf:ref}`post-grassmann`. It gives
$\{\psi_i,\psi_j\}=\{\bar\psi_i,\bar\psi_j\}
=\{\psi_i,\bar\psi_j\}=0$ by exterior multiplication.
The adjoint operators on the undoubled Hilbert exterior space instead
obey $\{\mathsf A(e_i),\mathsf C(e_j)\}=\delta_{ij}I$.
These formulas identify the barred integration symbols and the
adjoint contraction as different constructions from the same mode space.

The actual CAR algebra, positive vacuum state, replica law, and
completely positive recorded evolution are supplied by
{prf:ref}`thm-lqft-record-fock-reconstruction`,
{prf:ref}`thm-lqft-replica-isomorphism`, and
{prf:ref}`thm-lqft-record-car-channel`.
A finite Berezin action also requires its coefficient matrix and
integration orientation; its determinant identity is
{prf:ref}`def-fermionic-action`.

*Proof.* The faithful homomorphism and its inverse are
{ref}`(LQ.A1) <eq-fg-lq-a1>` and the coefficient expansion in its
orthonormal replica wedges. Taking the direct sum with $E^\vee$
and polarizing the defining squares gives the three Euclidean
anticommutators. The adjoint deletion calculation gives the mixed CAR.
Thus these are constructions and derived representation identities,
rather than a further assumption on the cloning transition.
In particular $\bar\psi_j\psi_i=-\psi_i\bar\psi_j$; interchanging the
indices instead produces a different monomial in general.
:::

:::{prf:remark} Endpoint and gauge conventions
:label: remark-sm-fermion-orientation

The source-based bilinear is $\bar\psi_iU_{ij}\psi_j$, with
$\bar\psi_i\mapsto\bar\psi_i\Omega_i^{-1}$. Its factors all contract at
the same endpoint. A target-based expression uses $U_{ji}$ and the conjugate
field at $j$. Reversing an oriented sum also requires transforming its
weights and links; equality of two actions is then an explicit reindexing
identity, rather than a consequence of Grassmann signs alone.
:::

:::{prf:definition} Finite fermionic action with comparison links
:label: def-sm-fermionic-action

For prescribed quadrature weights $a_{ij},b_{ij}$ and positive increments
$h_{ij}$, set

$$
S_{\mathrm f}=-\sum_{(i,j)\in E_{\mathrm{IG}}}
 a_{ij}\bar\psi_i\widetilde K_{ij}U_{ij}\psi_j
 -\sum_{(i,j)\in E_{\mathrm{CST}}}
 b_{ij}\bar\psi_i\frac{U_{ij}\psi_j-\psi_i}{h_{ij}},
\qquad \widetilde K=K-K^{\mathsf T}.
$$

Mass and Yukawa terms must use invariant contractions in the selected
representations. Gauge covariance of the displayed terms follows from the
comparison convention. The finite Berezin integral is the determinant of
the coefficient matrix, with the orientation and proof in
{prf:ref}`def-fermionic-action`.
:::

:::{prf:theorem} Temporal comparison and the trajectory derivative
:label: thm-sm-temporal-operator

For a smooth trajectory and field, suppose
$U_{ij}=I+h_{ij}\mathcal A(\dot\gamma(t_i))+O(h_{ij}^2)$ with uniform
second-order remainder. Then

$$
\frac{U_{ij}\psi_j-\psi_i}{h_{ij}}
 =\frac d{dt}\psi(\gamma(t_i))
  +\mathcal A(\dot\gamma(t_i))\psi(\gamma(t_i))+O(h_{ij}).
$$

A real assigned edge action defines a unitary phase directly. A node
potential difference yields pure-gauge links; equality with an integral of
fitness along a path is an additional relation. A proper-time choice for
$h_{ij}$ retains the trajectory and calibration conditions in
{prf:ref}`thm-fractal-faithful-embedding`.

Reflection positivity or unitary reconstruction of a field measure requires
that measure's transfer representation, such as the sufficient hypotheses
of {prf:ref}`lem-lqft-positive-transfer`. Neither an arbitrary QSD nor this
local Taylor formula supplies it. A directed forward-difference matrix
also need not be Hermitian or approach a Hermitian matrix at a population
sampling rate.
:::

:::{prf:proof}
Taylor expand $\psi(\gamma(t_j))$, multiply by the link expansion, and
divide the difference by $h_{ij}$, as in
{prf:ref}`thm-temporal-fermion-op`. For the last assertion, a nonzero directed
shift has an adjoint supported on the reverse edges. Their difference
need not shrink when more copies of the same directed stencil are added.
$\square$
:::

(sec-sm-structural-isomorphisms)=
### An explicit complex Clifford isomorphism

:::{prf:theorem} Clifford algebra of the specified spinor representation
:label: thm-sm-dirac-isomorphism

On $S=\Lambda^*\mathbb C^2$ use creation and contraction operators from
{prf:ref}`thm-dirac-structure-lqft`. The four operators

$$
\Gamma^{2j-1}=a_j+a_j^\dagger,\qquad
\Gamma^{2j}=i(a_j^\dagger-a_j),\qquad j=1,2,
$$

satisfy the Euclidean Clifford relations. Set
$\gamma^0=i\Gamma^1$, $\gamma^r=\Gamma^{r+1}$ for $r=1,2,3$.
Then the complex Clifford algebra for signature $(-,+,+,+)$ acts faithfully,
and

$$
\mathrm{Cl}_{\mathbb C}(\eta)\cong\operatorname{End}_{\mathbb C}(S)
\cong M_4(\mathbb C).
$$

Its algebra dimension is $16$; its irreducible spinor dimension is $4$.
This is a statement about the specified exterior representation, not an
isomorphism induced by skewness of $\widetilde K$.

On a supplied spin manifold with an orthonormal frame $e_a^{\ \mu}$,
$\gamma^\mu=e_a^{\ \mu}\gamma^a$ satisfies
$\{\gamma^\mu,\gamma^\nu\}=2g^{\mu\nu}I$.
A Dirac continuum limit additionally requires a spin connection, consistent
directional differences, quadrature convergence, and control of extra
lattice modes. The scalar kernel limit does not establish these conditions.
:::

:::{prf:proof}
The contraction identity
$a_i(e_j\wedge\omega)=\delta_{ij}\omega-e_j\wedge a_i\omega$
gives the CAR, and expansion gives the stated Clifford anticommutators.
The factor $i$ changes one generator's square to $-I$.
Clifford relations reorder every word into one of the $2^4$ ordered
monomials, so the abstract algebra has dimension at most $16$.

Conversely, $a_j$ and $a_j^\dagger$ are linear combinations of the
$\Gamma$ operators. The vacuum projector
$P_0=\prod_{j=1}^2(I-a_j^\dagger a_j)$ belongs to their algebra.
Products $a_I^\dagger P_0 a_J$, with the annihilators ordered to remove the
basis wedge $e_J$, are the matrix units sending $e_J$ to $e_I$ and all other
basis wedges to zero, up to a removable sign. They span all $16$ dimensions
of $\operatorname{End}(S)$. The induced surjective homomorphism is therefore
an isomorphism. Multiplying the frame coefficients in the anticommutator
gives $2e_a^{\ \mu}e_b^{\ \nu}\eta^{ab}I=2g^{\mu\nu}I$.

Equal fitnesses give $\widetilde K=0$, whose square cannot be a nonzero
Clifford metric. This explicit case shows why the score calculation alone
cannot supply the generators. $\square$
:::

### Component labels and generation copies

:::{prf:definition} Encoded component index and flavor multiplicity
:label: def-sm-flavor-index

The index $a=1,\ldots,d$ in the encoded viscous vector $c_i^{(a)}$ labels
components in a chosen coordinate basis. A generation index instead labels
copies of the same internal gauge representation $R$:

$$
\mathcal H_{\mathrm{int}}=R\otimes\mathbb C^{n_g},\qquad
\rho_{\mathrm{int}}(g)=\rho_R(g)\otimes I_{n_g}.
$$

A declaration $n_g=d$ specifies such a multiplicity space in a field model.
It is independent of identifying $n=d$ as a color rank.
:::

:::{prf:theorem} Multiplicity is independent of coordinate dimension
:label: thm-sm-generation-dimension

For an irreducible complex gauge representation $R$, gauge-commuting
endomorphisms of $R\otimes\mathbb C^{n_g}$ have the form
$I_R\otimes B$, with $B\in M_{n_g}(\mathbb C)$.
In particular the defining $SU(d)$ representation on $\mathbb C^d$ has one
copy of its representation, not $d$ invariant generation sectors.
Any positive integer $n_g$ is compatible with the same latent dimension.
If a supplied model chooses $n_g=d$, it has $d$ copies by construction.
:::

:::{prf:proof}
Write a commuting endomorphism in blocks $A_{\alpha\beta}:R\to R$.
Each block commutes with $\rho_R(g)$. Over $\mathbb C$, such a block has an
eigenvalue $\lambda$. Its nonzero eigenspace is invariant, so irreducibility
makes it all of $R$ and $A_{\alpha\beta}=\lambda I_R$. This proves the
commutant assertion. A nonzero invariant subspace of the defining $SU(d)$
representation contains one unit vector and hence all unit vectors by the
transitivity of $SU(d)$, so that representation is irreducible for $d\ge2$.
Finally tensoring with $\mathbb C^{n_g}$ changes no coordinate variable,
proving independence of $n_g$ and latent dimension. $\square$
:::

(sec-sm-scalar-sector)=
## 5. Scalar Consistency and Symmetry Breaking

:::{div} feynman-prose
A quartic potential is simple enough that we can find every stationary point
and every curvature by differentiation. This tells us the local restoring
forces. To obtain the Higgs mass matrix, we must also choose a complex
weak doublet and its covariant kinetic term. The extra choice matters:
a one-dimensional double well and a four-component Higgs field have different
sets of angular modes.
:::

:::{prf:definition} Scalar and doublet actions
:label: def-sm-scalar-action

A neutral real field uses the positive spatial action and vertex quadrature
of {prf:ref}`def-scalar-action`. A charged doublet instead uses specified
comparison links and

$$
S_H^{\mathrm{space}}
 =\frac12\sum_{\{i,j\}}c_{ij}\|U_{ij}H_j-H_i\|^2
   +\sum_i\mu_i V(H_i^\dagger H_i),\qquad c_{ij}\ge0,\quad\mu_i>0.
$$

The norm is independent of the chosen orientation because the links are
unitary. Under $H_i\mapsto\Omega_iH_i$, every term is gauge invariant.
A Lorentzian action uses the signed operator, normalization, and same-sample
quadrature assumptions in {prf:ref}`thm-cst-fractal-dalembertian-consistency`.
A sum of unsigned neighbor slopes is not a directional derivative without
specified directions and moment conditions.
:::

:::{prf:theorem} Spatial scalar consistency with the sampling density retained
:label: thm-sm-laplacian-convergence

Under {prf:ref}`assm-lqft-spatial-sampling`, the unnormalized operator

$$
L_{N,\epsilon}\phi(x)=\frac1{N\epsilon^{d+2}}
\sum_j k(d_{g_R}(x,X_j)^2/\epsilon^2)[\phi(X_j)-\phi(x)]
$$

has population limit

$$
\frac{m_2}{2}
 [\rho\Delta_{g_R}\phi+2\langle\nabla\rho,\nabla\phi\rangle].
$$

Its pointwise squared bias is $O(\epsilon^4)$ and its variance under the
identified joint full-gradient Poincare law is
$O((N\epsilon^{d+4})^{-1})$. The row-normalized density correction in
{prf:ref}`prop-density-corrected-limit` converges to $\Delta_{g_R}$ under
its additional density-estimation conditions. Same-sample scalar energies
can use {prf:ref}`lem-lqft-energy-sampling` with its two-particle and joint-law
bounds. Recorded IG operators must satisfy the stated comparison errors.

The length bandwidth $\epsilon$ is a reconstruction parameter here.
Setting it proportional to $\sqrt\tau$ does not alone verify the required
population, geometry, and sampling limits.

An alternative to the gradient estimate is the total-relative-entropy bound
in {prf:ref}`thm-mixing-variance-corrected`. If the actual joint law has
$H_N=\mathrm{KL}(\pi_N\Vert\rho^{\otimes N})<\infty$, the kernel summand
is bounded by $B_\epsilon=L_\phi\|\kappa\|_\infty\epsilon^{-d-1}$ and
that result gives

$$
\operatorname{Var}_{\pi_N}(L_{N,\epsilon}\phi(x))
\le\frac{4L_\phi^2\|\kappa\|_\infty^2
              (H_N+\tfrac12\log2)}{N\epsilon^{2d+2}}.
$$

This is another proved sufficient sampling estimate when its entropy and
bandwidth ratio tends to zero. If the one-particle marginal differs from
$\rho$, its mean error must also be included, as in
{prf:ref}`rem-lqft-existing-sampling-estimates`.
:::

:::{prf:proof}
In normal coordinates the quadratic term of
$[\phi(y)-\phi(x)]\rho(y)$ is
$\rho D^2\phi/2+\operatorname{sym}(\nabla\rho\otimes\nabla\phi)$.
Radial moments produce the two displayed density terms. The fourth-order
remainder and the Poincare estimate for the shrinking kernel are proved with
constants in {prf:ref}`thm-laplacian-convergence`. Exact density weighting
cancels $\rho$ from the numerator; row normalization fixes its zeroth
moment, as proved in {prf:ref}`prop-density-corrected-limit`. The energy
estimate uses pair quadrature, not pointwise convergence alone. $\square$
:::

:::{prf:theorem} Quartic normal form and the supplied Higgs mass matrix
:label: thm-sm-higgs-isomorphism

Suppose a deterministic order parameter $r\in\mathbb R^k$ has the
specified gradient normal form

$$
\dot r=a r-\alpha|r|^2r=-\nabla W(r),\qquad
W(r)=-\frac a2|r|^2+\frac\alpha4|r|^4,\qquad\alpha>0.
$$

For $a<0$ the origin is the unique minimum. For $a>0$ the minima satisfy
$|r|^2=a/\alpha$. At a minimum the Hessian is
$2\alpha r r^{\mathsf T}$: its radial eigenvalue is $2a$ and its $k-1$
angular eigenvalues are zero. The potential barrier from a minimum to the
origin is $a^2/(4\alpha)$.

For the additionally supplied electroweak doublet $H\in\mathbb C^2$ of
hypercharge $Y_H=1/2$, choose

$$
V(H)=-\mu^2H^\dagger H+\lambda(H^\dagger H)^2,
\qquad \mu^2,\lambda>0,
\qquad
D_\mu H=(\partial_\mu-ig_2\sigma^aW_\mu^a/2-ig_YB_\mu/2)H.
$$

With canonical field kinetic normalization and a chosen vacuum
$H_0=(0,v/\sqrt2)^{\mathsf T}$, one has

$$
v^2=\mu^2/\lambda,\qquad
m_h^2=2\mu^2,\qquad m_W^2=\frac{g_2^2v^2}{4},\qquad
m_Z^2=\frac{(g_2^2+g_Y^2)v^2}{4},\qquad m_A^2=0.
$$

The unbroken infinitesimal generator is $Q=T^3+Y$.
The first calculation reproduces the quartic calculation of
{prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts` when its normal
form applies. Applying it to Fractal Gas requires deriving that normal form
and its coefficients for its dynamics. Local Hessian curvature is a
linearized deterministic relaxation rate; a global stochastic spectral gap
or a quantum mass requires an additional operator identification.
:::

:::{prf:proof}
Differentiation gives
$\nabla W=(-a+\alpha|r|^2)r$ and
$D^2W=(-a+\alpha|r|^2)I+2\alpha rr^{\mathsf T}$.
The stationary points, their curvatures, and the barrier follow by
substitution. For $k=1$ the two minima give a pitchfork; for $k>1$ they form
a sphere with angular zero modes. Higher-order terms in a reduced model
must be estimated before using the exact quartic formulas.

Write the four real components of $H$ as $r/\sqrt2$. Then its potential
is the same polynomial with $a=\mu^2$ and $\alpha=\lambda$.
Substitute $H=(0,(v+h)/\sqrt2)^{\mathsf T}$ into $V$.
The linear term vanishes at $v^2=\mu^2/\lambda$, and the quadratic term
is $\mu^2h^2=\tfrac12m_h^2h^2$.
At the constant vacuum, direct multiplication gives

$$
\|D_\mu H_0\|^2
 =\frac{v^2}{8}\left[g_2^2((W_\mu^1)^2+(W_\mu^2)^2)
                   +(g_2W_\mu^3-g_YB_\mu)^2\right].
$$

Set $W^\pm=(W^1\mp iW^2)/\sqrt2$ and

$$
Z=\frac{g_2W^3-g_YB}{\sqrt{g_2^2+g_Y^2}},\qquad
A=\frac{g_YW^3+g_2B}{\sqrt{g_2^2+g_Y^2}}.
$$

The quadratic form becomes $m_W^2W^+W^-+\tfrac12m_Z^2Z^2$, with no
$A^2$ term. Finally $(T^3+Y)H_0=0$. These calculations determine the mass
matrix of this specified classical field action. $\square$
:::

(sec-sm-completing)=
## 6. Internal Representations and Recorded Operators

:::{div} feynman-prose
The internal charge table can be constructed without guessing its dimension.
Start with five basis vectors, three carrying color and two carrying weak
isospin. Form their even exterior products. Counting and decomposing these
products gives sixteen states with the familiar charges. This proves a
representation statement for the chosen five-dimensional space.

The same table lets us check the conventional gauge anomalies explicitly.
Every color and weak multiplicity enters the charge sums; their contributions
cancel within one generation, and the weak-doublet count is even. These checks
establish the stated anomaly cancellations for this representation. Identifying
the representation with recorded dynamics still uses the measure and generator
maps developed here.
:::

### A direct Spin(10) branching calculation

:::{prf:theorem} The chosen half-spin representation and one charge generation
:label: thm-sm-so10-isomorphism

Let $E=\mathbb C^3\oplus\mathbb C^2$ and choose the traceless hypercharge
operator

$$
Y=\operatorname{diag}(-1/3,-1/3,-1/3,1/2,1/2).
$$

The positive half-spin representation of $\mathrm{Spin}(10)$ can be
realized as $S_+=\Lambda^{\mathrm{even}}E$, of complex dimension $16$.
Under the subgroup $S(U(3)\times U(2))\subset SU(5)$ it decomposes as

$$
S_+=(3,2)_{1/6}\oplus(\bar3,1)_{-2/3}
\oplus(\bar3,1)_{1/3}\oplus(1,2)_{-1/2}
\oplus(1,1)_1\oplus(1,1)_0.
$$

These are the conventional left-handed fields
$Q,u^c,d^c,L,e^c,\nu^c$ of one generation including a neutral singlet.
The spin representation is a representation of $\mathrm{Spin}(10)$;
it does not descend to $SO(10)$ because the central element $-1$ acts as
minus the identity. Choosing this internal field space realizes the stated
branching. Dimension counting of $(x,v)$ or the CST does not identify it
with the recorded walker state space.
:::

:::{prf:proof}
Use five creation-contraction pairs on $\Lambda^*E$. Their ten Clifford
generators were constructed in {prf:ref}`thm-dirac-structure-lqft`.
Products of an even number of generators preserve exterior parity and
supply the spin action. Matrix units between wedges of the same parity,
constructed as in {prf:ref}`thm-sm-dirac-isomorphism`, belong to the even
Clifford algebra. They show irreducibility on each parity sector.
Its central element $-1$ acts as $-I$.
The even sector has dimension $\binom50+\binom52+\binom54=1+10+5=16$.
The $SU(5)$ action on $E$, and hence on its exterior algebra, is the
restriction of this spin action: its infinitesimal operators are the
number-preserving bilinears $\sum A_{ij}a_i^\dagger a_j$ with
$\operatorname{Tr}A=0$, which are the corresponding Clifford bivectors.

Put $C=(3,1)_{-1/3}$ and $W=(1,2)_{1/2}$. The identity
$\Lambda^r(C\oplus W)=\bigoplus_{p+q=r}\Lambda^pC\otimes\Lambda^qW$
gives

$$
\Lambda^2E=(\bar3,1)_{-2/3}\oplus(3,2)_{1/6}\oplus(1,1)_1,
$$

because $\Lambda^2\mathbb C^3\cong\overline{\mathbb C^3}$ through the
invariant volume form and $\Lambda^2\mathbb C^2\cong\mathbb C$.
Similarly

$$
\Lambda^4E=(\bar3,1)_{1/3}\oplus(1,2)_{-1/2}.
$$

The first summand uses $p=q=2$ and the second $p=3,q=1$.
Hypercharges add in exterior products; $\Lambda^0E$ is the neutral singlet.
This proves the full decomposition, not just its dimension.

For the global group, the map

$$
(A,B,z)\longmapsto\operatorname{diag}(z^{-2}A,z^3B)
$$

from $SU(3)\times SU(2)\times U(1)$ onto $S(U(3)\times U(2))$
has kernel $z^6=1$, $A=z^2I_3$, $B=z^{-3}I_2$.
Surjectivity follows by choosing $z$ from either block determinant; the
other is fixed by the determinant-one condition. This gives the quotient
by $\mathbb Z_6$. Its action on $S_+$ is faithful: an element acting
trivially on $\Lambda^4E\cong E^*$ acts trivially on $E$.
$\square$
:::

:::{prf:proposition} Local anomaly coefficients of the stated generation
:label: prop-sm-generation-anomaly-cancellation

For the left-handed representations in {prf:ref}`thm-sm-so10-isomorphism`,
the four-dimensional perturbative gauge and mixed gauge-gravitational anomaly
coefficients vanish. Normalize the cubic color index of $3$ to one and the
quadratic index of $3$ and $2$ to $1/2$. The usual mod-two $SU(2)$ doublet
count is even. These conclusions hold generation by generation, including
the neutral singlet, and hence for every $n_g$.

*Proof.* Antifundamentals have opposite cubic color index and equal quadratic
index. Thus the color-cubic coefficient and the two mixed coefficients are

$$
\begin{aligned}
\mathcal A_{333}&=2-1-1=0,\\
\mathcal A_{33Y}&=2\left(\frac16\right)\frac12
 +\left(-\frac23\right)\frac12+\left(\frac13\right)\frac12=0,\\
\mathcal A_{22Y}&=3\left(\frac16\right)\frac12
 +\left(-\frac12\right)\frac12=0.
\end{aligned}
$$

For completeness, the charge and cubic-charge traces, with all internal
multiplicities included, are

$$
\begin{aligned}
\mathcal A_{\mathrm{grav}\,Y}
&=6\left(\frac16\right)+3\left(-\frac23\right)
 +3\left(\frac13\right)+2\left(-\frac12\right)+1+0=0,\\
\mathcal A_{YYY}
&=6\left(\frac16\right)^3+3\left(-\frac23\right)^3
 +3\left(\frac13\right)^3+2\left(-\frac12\right)^3+1+0\\
&=\frac1{36}-\frac89+\frac19-\frac14+1=0.
\end{aligned}
$$

A coefficient with one nonabelian generator and two hypercharges vanishes
because the nonabelian generator is traceless. Mixed coefficients involving
both distinct simple factors similarly contain a trace of a single traceless
generator. For the weak doublet,
$\{T^b,T^c\}=\delta^{bc}I/2$, so
$\operatorname{Tr}(T^a\{T^b,T^c\})=0$; this proves the weak-cubic coefficient
vanishes. Singlets add zero. The four left-handed weak doublets consist of
three color copies of $Q$ and one $L$, giving $3+1=4$ per generation. The
usual $SU(2)$ global-anomaly sign for doublets is $(-1)^{4n_g}=1$.
$\square$

The trace coefficients above use the four-dimensional chiral anomaly criterion
reviewed in [Bilal, *Lectures on Anomalies*](https://arxiv.org/abs/0802.0634),
and the doublet sign uses
[Witten's $SU(2)$ anomaly calculation](https://www.sciencedirect.com/science/article/pii/0370269382907286).
This verifies these local coefficients and the ordinary doublet parity test
on spin backgrounds. It neither selects $n_g$ nor establishes every possible
global anomaly statement for arbitrary bundles of the quotient gauge group.
The finite Berezin invariance in {prf:ref}`def-sm-total-action` and its continuum
chiral realization remain distinct calculations.
:::

This exterior-algebra construction and the distinction between the spin
group and its orthogonal quotient are developed in
[Baez and Huerta, *The Algebra of Grand Unified Theories*](https://arxiv.org/abs/0904.1556).
The representation calculation fixes the chosen charge content; its dynamics
and any unification scale are additional questions.

### Walker-role observables

:::{prf:definition} Recorded walker-role partition
:label: def-sm-walker-role-partition

At a frame $t$, let $A_t\subseteq\{1,\ldots,N\}$ be the alive set,
$c_c(i,t)$ the recorded clone-companion index, $F_i(t)$ its fitness, and
$\mathbf1_{\mathrm{clone}}(i,t)$ its cloning indicator. Use valid companion
indices, or the same declared index-clamping convention as the recorded
analysis. Define

$$
\begin{aligned}
\Delta_t&=\{i\in A_t:\mathbf1_{\mathrm{clone}}(i,t)=1\},\\
\mathrm{SR}_t&=\{i\in A_t\setminus\Delta_t:
                    \exists j\in\Delta_t,\ c_c(j,t)=i\},\\
\mathrm{WR}_t&=\{i\in A_t\setminus(\Delta_t\cup\mathrm{SR}_t):
                          F_{c_c(i,t)}(t)>F_i(t)\},\\
\mathrm P_t&=A_t\setminus(\Delta_t\cup\mathrm{SR}_t\cup\mathrm{WR}_t).
\end{aligned}
$$

These are the delta, strong-resister, weak-resister, and persister roles.
Set $L_t=\Delta_t\cup\mathrm{SR}_t$ and
$R_t=\mathrm{WR}_t\cup\mathrm P_t$.
The letters denote recorded roles; identifying them with eigenspaces of a
Dirac chirality operator requires another map.
:::

:::{prf:proposition} Partition and same-frame companion constraint
:label: prop-sm-walker-role-partition

The four role sets are pairwise disjoint and cover $A_t$.
Moreover, for every $i\in\Delta_t$, its companion cannot lie in $R_t$.
:::

:::{prf:proof}
Each successive set is formed inside the complement of its predecessors,
and the final complement exhausts $A_t$. If the companion of a delta is
alive, it is either itself a delta or, by being targeted, belongs to
$\mathrm{SR}_t$. In both cases it lies in $L_t$. If it is dead, it lies in
neither $L_t$ nor $R_t$. $\square$
:::

:::{prf:definition} Role chirality and baseline observables
:label: def-sm-walker-chirality

For $N>0$, set $\chi_i=1$ on $L_t$, $\chi_i=-1$ on $R_t$, and
$\chi_i=0$ outside $A_t$. Preserve the recorded normalizations

$$
\chi_{\mathrm{mean}}=\frac1N\sum_i\chi_i,
\qquad f_L=\frac{|L_t|}{N},\qquad
f_{\Delta\to R}=
 \frac{\sum_{i\in\Delta_t}\mathbf1_{\{c_c(i,t)\in R_t\}}}{|\Delta_t|},
$$

and

$$
M_{LR}=\frac1{N_{\Delta\to R}}
 \sum_{\substack{i\in\Delta_t\\c_c(i,t)\in R_t}}
 e^{i(F_{c_c(i,t)}-F_i)/\hbar_{\mathrm{eff}}},\qquad
N_{\Delta\to R}=\sum_{i\in\Delta_t}\mathbf1_{\{c_c(i,t)\in R_t\}}.
$$

A quotient with zero count is assigned zero, as in the recorded code.
Then the exact same-frame identities are

$$
\chi_{\mathrm{mean}}=2f_L-|A_t|/N,
\qquad N_{\Delta\to R}=f_{\Delta\to R}=M_{LR}=0.
$$

Dead walkers contribute zero to numerators but still enter the fixed $N$
normalization. A nontrivial cross-role observable would require a different
pair selection, temporal comparison, or classification; it would be a
different statistic from the one defined here.
:::

:::{prf:theorem} Recorded roles, spinor bilinears, and scalar phase proxies
:label: thm-sm-ew-operator-layers

The recorded analysis contains distinct algebraic constructions:

1. The role statistics above are functions of the same-frame classification.
   Their delta-to-right values are identically zero under these definitions.
2. Given supplied four-component complex vectors $\psi_i$, Clifford
   matrices, and projectors $P_{L,R}=(I\mp\gamma^5)/2$, bilinears
   $\bar\psi_i\Gamma P_{L,R}\psi_j$ are defined. The vector identity
   $J_V^\mu=J_L^\mu+J_R^\mu$ follows from $P_L+P_R=I$.
   Applying an additional recorded $L/R$ pair mask does not turn that mask
   into a spinor projector.
3. A scalar phase $u_{ij}$ gives the implemented real statistic
   $\operatorname{Re}[u_{ij}\psi_i^\dagger\widehat\gamma^0
                                      \Gamma P\psi_j]$.
   A full nonabelian gauge bilinear instead requires an internal doublet
   index and a matrix link between its endpoints.

The code's $\widehat\gamma^\mu$ uses signature $(+,-,-,-)$, whereas the
geometric convention above is $(-,+,+,+)$. Multiplication of all generators
by $i$ exchanges these Clifford signatures and leaves $\gamma^5$ unchanged.
The numerical Dirac adjoint uses its declared $\widehat\gamma^0$.

The implementation named `compute_su2_gauge_link` returns the scalar phase

$$
u_{ij}^{(2)}=\exp\!\left[
 \frac{i\pi}{2h}\frac{|F_j-F_i|}{|F_j-F_i|+\varepsilon}\right],
\qquad h>0,\quad\varepsilon>0.
$$

Its magnitude is one, but $u_{ji}^{(2)}=u_{ij}^{(2)}$ generally differs from
$(u_{ij}^{(2)})^{-1}$. It is therefore a scalar proxy, not the $SU(2)$
comparison-link construction of {prf:ref}`thm-sm-su2-emergence`.
A scalar multiple $uI_2$ has determinant $u^2$ and is in $SU(2)$ only when
$u^2=1$.
:::

:::{prf:proof}
The role identities follow from {prf:ref}`prop-sm-walker-role-partition`.
The Clifford calculation gives $(\gamma^5)^2=I$, so the two projectors sum
to $I$; linearity gives the current identity. Internal gauge transformations
commute with spinor matrices. Consequently the correctly contracted
$\bar\psi_i\Gamma P U_{ij}\psi_j$ is invariant under the endpoint
transformations when its internal indices and representations match.

The functions `classify_walkers_vectorized` and the left-right calculation
in `src/fragile/physics/electroweak/chirality.py` implement the displayed
partition and mask. The scalar phase and real bilinear are implemented in
`src/fragile/physics/electroweak/electroweak_spinors.py`. Their modulus,
reversal, and determinant assertions follow directly from the displayed
exponential. These are exact descriptions of those computations. A fit to
a correlator of these proxies needs an independent physical interpretation;
assigning a particle name does not prove its spectral channel.
$\square$
:::

Further details of these recorded formulas appear in
{prf:ref}`prop-qft-ew-chirality-realization` and
{prf:ref}`prop-qft-ew-spinor-realization`. Their use in
{prf:ref}`thm-qft-ew-active-pipeline` must retain the distinction between
role statistics, supplied spinor algebra, and scalar phase proxies.

(sec-sm-coupling-matching)=
## 7. Coupling Proxies, Mixing, and Mass Models

:::{div} feynman-prose
A link contains the product of a coupling and a field. Rescaling the field
and dividing the coupling by the same factor leaves that link unchanged.
To measure a coupling we therefore need a field normalization, usually
fixed by the kinetic action. A kernel width or a phase variance can be a
reproducible statistic, but its conversion to a physical coupling needs
that normalization and a matching calculation.
:::

### Dimensionless statistics and field normalization

:::{prf:definition} Interaction-range coupling proxies
:label: def-sm-coupling-definition

Fix reference scales $\ell_0,t_0,m_0>0$ and define

$$
\widehat\epsilon=\epsilon/\ell_0,\qquad
\widehat\hbar=\hbar_{\mathrm{eff}}t_0/(m_0\ell_0^2),\qquad
\widehat\nu=\nu t_0,
$$

when $\nu$ has inverse-time units and the viscous kernel is dimensionless.
Other kernel units require the corresponding reference factor.
For a specified probability law $\mu$ on recorded pairs define the
nonnegative statistics

$$
\begin{aligned}
\widehat g_1^2
 &=\frac{\widehat\hbar}{\widehat\epsilon_d^2}\mathcal N_1,
&\quad \mathcal N_1&=\mathbb E_\mu
                 e^{-d_{\mathrm{alg}}^2/\epsilon_d^2},\\
\widehat g_2^2
 &=\frac{2\widehat\hbar}{\widehat\epsilon_c^2}
                       \frac{C_2(2)}{C_2(n)},
& C_2(n)&=\frac{n^2-1}{2n},\qquad n\ge2,\\
\widehat g_n^2
 &=\frac{\widehat\nu^2}{\widehat\hbar^2}
      \frac{n(n^2-1)}{12}\mathbb E_\mu K_{\mathrm{visc}}^2.
\end{aligned}
$$

These preserve the chapter's proposed range, Casimir, and kernel-moment
parameterizations as defined proxies. Their normalization factors are
choices. The joint law $\mu$ must be specified; a QSD label means the
identified quasi-stationary law, not an invariant or time-history law.
Its moments generally depend on the potential and the full parameter set.
Physical couplings $g_Y,g_2,g_3$ instead refer to canonically normalized
field actions. Use $g_1=\sqrt{5/3}\,g_Y$ when quoting the conventional
unified hypercharge normalization.
:::

:::{prf:theorem} Bounds and identifiability of the diversity proxy
:label: thm-sm-g1-coupling

For the specified pair law, $0<\mathcal N_1\le1$ when distances are finite
almost surely. If $\mu\{d_{\mathrm{alg}}\le\epsilon_d\}\ge p$, then
$\mathcal N_1\ge p/e$. Hence

$$
\frac{p\widehat\hbar}{e\widehat\epsilon_d^2}
 \le\widehat g_1^2\le
 \frac{\widehat\hbar}{\widehat\epsilon_d^2}.
$$

The lower bound is absent without control of the sampled pair distances.
A phase or a companion amplitude alone does not identify a canonically
normalized $U(1)$ coupling.
:::

:::{prf:proof}
The kernel lies in $(0,1]$ and is at least $e^{-1}$ on the stated event.
Integrating proves the bounds. A link of the form
$\exp(i g\int A)$ is unchanged by $A\mapsto cA$, $g\mapsto g/c$ for any
nonzero real $c$. A kinetic normalization or another field-scale constraint
is therefore needed to distinguish these couplings. $\square$
:::

:::{prf:theorem} Casimir normalization of the chosen doublet proxy
:label: thm-sm-g2-coupling

For a Hermitian traceless basis $T^a$ of the defining $SU(n)$ representation
with $\operatorname{Tr}(T^aT^b)=\delta^{ab}/2$,

$$
\sum_{a=1}^{n^2-1}T^aT^a=\frac{n^2-1}{2n}I.
$$

This proves the Casimir factor used in $\widehat g_2$. At $n=3$,
$C_2(2)/C_2(3)=9/16$ and
$\widehat g_2^2=9\widehat\hbar/(8\widehat\epsilon_c^2)$.
The ratio is a normalization in that proxy. The independent gauge factors
$SU(2)$ and $SU(n)$ in {prf:ref}`cor-sm-gauge-group` do not impose this
ratio on their physical couplings.
:::

:::{prf:proof}
Conjugation by a unitary matrix acts orthogonally on the traceless Hermitian
basis in the trace inner product. Therefore $\sum_aT^aT^a$ commutes with
all unitaries and is scalar. Its trace is $(n^2-1)/2$, giving the factor
$(n^2-1)/(2n)$. Substitution of $n=2,3$ gives the stated numbers.
Independent nonnegative coefficients for the two gauge kinetic terms
preserve both gauge symmetries, so symmetry alone does not fix their ratio.
$\square$
:::

:::{prf:theorem} Viscous-force moments and the kernel proxy
:label: thm-sm-g3-coupling

For finite second moments, the actual force statistic obeys

$$
\mathbb E\|F_i^{\mathrm{visc}}\|^2
=\nu^2\sum_{j,k}\mathbb E
 [K_{ij}K_{ik}(v_j-v_i)\cdot(v_k-v_i)].
$$

If $K_{ij}\ge0$, then pointwise

$$
\|F_i^{\mathrm{visc}}\|^2
\le\nu^2\left(\sum_jK_{ij}\right)
            \left(\sum_jK_{ij}\|v_j-v_i\|^2\right).
$$

The proxy $\widehat g_n$ in {prf:ref}`def-sm-coupling-definition` exists
when its kernel second moment is finite. Its factor $n(n^2-1)/12$ is an
assigned normalization; it does not follow from the adjoint dimension
$n^2-1$ alone. Identifying it with a force statistic additionally requires
control of the velocity moments and cross terms in the displayed identity.
:::

:::{prf:proof}
Expand the squared norm of the sum to obtain the exact double sum.
Weighted Cauchy--Schwarz gives the bound. If all velocities coincide, the
force is zero even when every $K_{ij}$ is positive. This example rules out
an identity replacing the full force moment by a positive constant times
$\nu^2\mathbb E K_{\mathrm{visc}}^2$ for general recorded states.
$\square$
:::

:::{prf:proposition} What coupling matching would establish
:label: prop-sm-coupling-correspondence

Let $p$ denote an explicitly chosen list of algorithm and reconstruction
parameters. A matching map $p\mapsto(g_1,g_2,g_3)$ requires canonical field
normalizations and equality of specified field observables or effective-action
coefficients. If a proposed map is continuously differentiable between
three-dimensional open parameter domains and its Jacobian determinant is
nonzero at $p_0$, it is locally invertible there. A global bijection requires
additional injectivity and range information. The proxy definitions alone
provide no such conclusion.
:::

:::{prf:proof}
The local conclusion is the inverse function theorem under its stated
hypotheses. The proxy statistics depend on moments of an additional law and
on more than three available algorithm parameters; their definition does
not establish a three-dimensional injective map. Even a locally invertible
map can fail to be globally one-to-one, so the last requirements are
separate. $\square$
:::

### Conditional perturbative running and unification

:::{prf:corollary} One-loop running for the supplied Standard Model field content
:label: cor-sm-beta-functions

Suppose the continuum field model has the conventional three generations
of {prf:ref}`thm-sm-so10-isomorphism`, one complex Higgs doublet, canonical
kinetic terms, and the usual four-dimensional perturbative gauge theory
with modified-minimal-subtraction renormalization. At scales where this
field content is active, its one-loop coefficients are

$$
\beta(g_1)=\frac{41}{10}\frac{g_1^3}{16\pi^2},\qquad
\beta(g_2)=-\frac{19}{6}\frac{g_2^3}{16\pi^2},\qquad
\beta(g_3)=-7\frac{g_3^3}{16\pi^2},
\qquad g_1=\sqrt{5/3}\,g_Y.
$$

These are coefficients of that specified perturbative theory. Applying them
to recorded Fractal Gas proxies requires the continuum field and coupling
matching hypotheses; changing a companion bandwidth is not by itself this
renormalization-group flow.
:::

:::{prf:proof}
Use the standard one-loop gauge coefficient

$$
b_0=\frac{11}{3}C_2(G)
 -\frac23\sum_{\text{Weyl}}T(R_f)
 -\frac13\sum_{\text{complex scalar}}T(R_s),
\qquad \beta(g)=-b_0g^3/(16\pi^2).
$$

For color, each generation contributes $2$ to the Weyl index sum and the
Higgs contributes zero, giving $b_3=11-(2/3)6=7$.
For weak isospin, each generation again contributes $2$ and the complex
Higgs contributes $1/2$, giving $b_2=22/3-4-1/6=19/6$.
For the hypercharge convention $Q=T^3+Y$, one generation contributes

$$
\sum_fY_f^2
=6(1/6)^2+3(2/3)^2+3(1/3)^2+2(1/2)^2+1=10/3.
$$

The Higgs contributes $2(1/2)^2=1/2$, so $b_Y=-20/3-1/6=-41/6$.
Rescaling $g_1=\sqrt{5/3}g_Y$ changes $41/6$ to $41/10$.
This proves the representation sums conditional on the perturbative loop
formula; it is not a derivation of that loop formula from the particle
transition kernel. For general color rank and $n_g$ copies with two color
Dirac flavors each, the color coefficient is $(11n-4n_g)/3$, whose sign
requires $11n>4n_g$. $\square$
:::

The coefficients and normalization are recorded, for example, in the
[Standard Model part of Appendix B of Fox, Kribs, and Martin](https://link.aps.org/accepted/10.1103/PhysRevD.90.075006).
They must be distinguished from the running in that paper's additional
matter sectors and from a finite-record statistical proxy.

:::{prf:proposition} Conditional unification and the Weinberg angle
:label: prop-sm-unification

If canonically normalized couplings satisfy
$g_1=g_2=g_3=g_G$ at a supplied unification scale, with
$g_1=\sqrt{5/3}g_Y$, then

$$
\sin^2\theta_W=\frac{g_Y^2}{g_Y^2+g_2^2}=\frac38.
$$

If the proxies $\widehat g_1,\widehat g_2$ have also been matched to
$g_1,g_2$ at that scale, equality imposes

$$
\frac{\widehat\epsilon_d^2}{\widehat\epsilon_c^2}
 =\frac{\mathcal N_1 C_2(n)}{2C_2(2)}.
$$

This is an algebraic constraint at an assumed meeting point. It does not
establish that such a point exists or that a physical coupling matching
has been achieved.
:::

:::{prf:proof}
The hypercharge normalization gives $g_Y^2=3g_G^2/5$; substitution gives
$3/8$. Equating the two proxy definitions and cancelling their common
$\widehat\hbar$ gives the range relation. $\square$
:::

### CP transformations and unitary mixing

:::{div} feynman-prose
Noncommuting updates do not automatically violate CP. CP is a specified
transformation of states and fields, and symmetry asks whether the whole law
is invariant under that transformation. An observable that changes sign
under CP gives a clean test: its expectation must vanish in a CP-invariant
law. This lets us state a usable test without guessing a phase from the
mismatch of two kernel widths.
:::

:::{prf:definition} A CP comparison on a field ensemble
:label: def-sm-cp-transformation

Specify an involution $\Theta$ on the selected record-and-field ensemble.
Its spatial part sends $(x,v)$ to $(-x,-v)$ on a reflection-invariant domain;
its internal charge-conjugation part sends a representation to its complex
conjugate, including conjugation of the chosen link and matter data.
Any parity action on vector and spinor components must be included in that
model. CP invariance means $\Theta_*\mu=\mu$ for the actual law $\mu$.

Reversing an oriented edge replaces a comparison link by its inverse. This
operation alone is not a definition of charge conjugation for arbitrary
nonabelian representations. Reversing CST time order would compare with a
reversed record and concerns time reversal, not the CP operation just
defined. A finite directed record need not be fixed by either comparison.
:::

:::{prf:theorem} A direct CP-odd expectation criterion
:label: thm-sm-cp-violation

For an integrable observable $J$ with $J\circ\Theta=-J$,
$\Theta_*\mu=\mu$ implies $\mathbb E_\mu J=0$.
Thus a nonzero expectation proves failure of CP invariance for that
specified transformation and law. A zero expectation of one observable is
not sufficient to prove CP invariance.

For a Markov field model, equivariance
$P(\Theta s,\Theta B)=P(s,B)$ and an invariant initial distribution imply
CP invariance at every time. Unequal real radial kernel widths do not by
themselves contradict this equivariance. In particular Gaussian convolution
operators of different widths are real, parity equivariant, and commute.
A product of real-valued phase angles has identically zero imaginary part;
it cannot serve as a nonzero CP-odd phase invariant.
:::

:::{prf:proof}
Change variables under $\Theta$ to obtain
$\mathbb E_\mu J=\mathbb E_\mu(J\circ\Theta)=-\mathbb E_\mu J$.
For the Markov statement, pushing a measure through one transition commutes
with $\Theta_*$ by the kernel identity; induction preserves invariance.
Gaussian convolution commutes because convolution is associative and
commutative, and an even real kernel is parity equivariant. These statements
hold for different widths. Finally a product of real numbers is real.
Noncommutativity of other update operators, when present, is a separate
statement from their equivariance under $\Theta$. $\square$
:::

:::{prf:proposition} A rephasing-invariant phase statistic and a conditional bound
:label: prop-sm-cp-magnitude

For a supplied unitary mixing matrix $V$, choose distinct rows $a,b$ and
columns $i,j$ and set

$$
J_{ab;ij}=\operatorname{Im}
 [V_{ai}V_{bj}\overline{V_{aj}}\,\overline{V_{bi}}].
$$

This statistic is invariant under independent row and column rephasings,
changes sign under complex conjugation, and satisfies $|J_{ab;ij}|\le1/4$.
For a differentiable family $J(s,t)$ whose independently verified exchange
symmetry is $J(s,t)=-J(t,s)$, a bound $|\partial_1J|\le L$ implies
$|J(s,t)|\le L|s-t|$. Taking $s=\epsilon_d^2,t=\epsilon_c^2$ is possible
only if that exchange symmetry has been proved for the chosen family.
It does not follow from the widths themselves.
:::

:::{prf:proof}
In the quartet every row and column phase occurs once with each sign and
cancels. Complex conjugation negates its imaginary part. Row normalization
gives $|V_{ai}V_{aj}|\le(|V_{ai}|^2+|V_{aj}|^2)/2\le1/2$ and similarly
for row $b$, proving the bound. Antisymmetry gives $J(t,t)=0$; integrating
$\partial_1J$ from $t$ to $s$ gives the last inequality. $\square$
:::

:::{prf:corollary} Unitary mixing from two orthonormal flavor bases
:label: cor-sm-ckm-matrix

For two orthonormal eigenbases $U_u,U_d\in U(n_g)$ of supplied flavor mass
operators, $V=U_u^\dagger U_d$ is unitary. With nondegenerate Dirac masses
and generic mixing, its physically distinct parameters under row and
column rephasing consist of $n_g(n_g-1)/2$ angles and
$(n_g-1)(n_g-2)/2$ phases. This gives one such phase for $n_g=3$.
A probability-conserving transition matrix or an average of transition
amplitudes need not be unitary, so it cannot replace these basis hypotheses.
:::

:::{prf:proof}
$V^\dagger V=U_d^\dagger U_uU_u^\dagger U_d=I$.
The real dimension of $U(n_g)$ is $n_g^2$. Row and column phases remove
$2n_g-1$ independent parameters because a common opposite rephasing acts
trivially. Of the remaining $(n_g-1)^2$ parameters,
$n_g(n_g-1)/2$ are real rotation angles; subtraction gives the stated
phase count. At degeneracies or vanishing mixing angles stabilizers can
increase and the generic count changes. For comparison,
$\left(\begin{smallmatrix}1/2&1/2\\1/2&1/2\end{smallmatrix}\right)$
preserves probabilities but has rank one and is not unitary. $\square$
:::

### Ancestry, Majorana terms, and the seesaw calculation

:::{prf:definition} Ancestral pullback
:label: def-sm-ancestral-reflection

Choose a parent map $p(i)$ from explicitly recorded ancestry, fixing roots.
When it is single valued, the pullback on scalar episode fields is
$(\mathcal Rf)_i=f_{p(i)}$. Multiple parent conventions require a specified
choice or weighted map. For internal fields, include a comparison transport
from the parent's fiber to the child's fiber.

This map is generally many-to-one and $\mathcal R^2f_i=f_{p(p(i))}$ need
not equal $f_i$. The retained name "ancestral reflection" therefore denotes
a pullback, not an involution. It contains no spinor chirality exchange or
charge conjugation unless those operations are separately defined.
:::

:::{prf:theorem} Gauge criterion for a chosen Majorana bilinear
:label: thm-sm-majorana-mass

Let $\chi_a$ be two-component Weyl Grassmann fields and
$\varepsilon_{\alpha\beta}$ the antisymmetric spinor contraction. Then
$B_{ab}=\chi_a^\alpha\varepsilon_{\alpha\beta}\chi_b^\beta$ is symmetric
in the internal labels $a,b$. A Majorana term has the form

$$
S_M=\frac12\sum_{a,b}M_{ab}B_{ab}+\text{conjugate term},\qquad
M^{\mathsf T}=M.
$$

It is gauge invariant precisely when
$R(g)^{\mathsf T}MR(g)=M$ for its chosen internal representation.
A single field of nonzero unbroken $U(1)$ charge has no such nonzero
constant mass term. A neutral singlet can have one.

An ancestral kernel can be used to define coefficients, for example the
specified energy-scale ansatz

$$
h_{ij}=\frac{\hbar_{\mathrm{eff}}}{\Delta t_{ij}}
       \exp[-|\Phi_i-\Phi_j|/\Phi_0],\qquad
\Phi_0>0,\quad\Delta t_{ij}>0,
$$

on selected parent-child pairs. Its symmetric, gauge-compatible part gives
a finite nonlocal bilinear. Identifying a local continuum Majorana mass
requires localization, spinor transport, and action normalization estimates.
Neither ancestry nor this assigned exponential proves that the algorithm
produces that mass coefficient.
:::

:::{prf:proof}
Interchange $a,b$, anticommute the two Grassmann generators, and interchange
the spinor indices. The Grassmann sign cancels the antisymmetry of
$\varepsilon$, giving $B_{ba}=B_{ab}$. The antisymmetric part of $M$
therefore contributes zero. Substitution of $\chi\mapsto R(g)\chi$ gives
the stated invariance criterion. For charge $q$, it becomes
$e^{2iq\alpha}M=M$ for every $\alpha$, forcing $M=0$ when $q\ne0$.
A scalar neutral representation has no such obstruction. The ancestral
formula defines a coefficient with energy units and the stated suppression;
its local mass interpretation is a separate limiting statement. $\square$
:::

:::{prf:proposition} Exact two-state seesaw and its hierarchy condition
:label: prop-sm-seesaw

For the real symmetric neutral mass matrix

$$
\mathcal M=\begin{pmatrix}0&m_D\\m_D&M\end{pmatrix},\qquad M>0,
$$

its eigenvalues are $(M\pm\sqrt{M^2+4m_D^2})/2$.
The absolute value of the light eigenvalue satisfies

$$
m_{\mathrm{light}}
=\frac{\sqrt{M^2+4m_D^2}-M}{2},\qquad
0\le\frac{m_D^2}{M}-m_{\mathrm{light}}
\le\frac{m_D^4}{M^3}.
$$

Thus $|m_D|\ll M$ gives the seesaw regime
$m_{\mathrm{light}}=m_D^2/M+O(m_D^4/M^3)$.
At fixed $m_D\ne0$, reducing $M$ increases the light eigenvalue's magnitude;
exponentially suppressing the heavy Majorana coefficient alone does not
produce this hierarchy. An unbroken electrically charged lepton cannot
receive a Majorana term of the preceding neutral-singlet form.
:::

:::{prf:proof}
Solve $\lambda^2-M\lambda-m_D^2=0$. The positive magnitude of the
negative root obeys $m_{\mathrm{light}}(M+m_{\mathrm{light}})=m_D^2$.
Hence $m_{\mathrm{light}}\le m_D^2/M$ and
$m_D^2/M-m_{\mathrm{light}}=m_{\mathrm{light}}^2/M\le m_D^4/M^3$.
Differentiation gives
$\partial_Mm_{\mathrm{light}}=(M/\sqrt{M^2+4m_D^2}-1)/2<0$ for
$m_D\ne0$. Gauge exclusion for a charged field follows from the preceding
theorem. $\square$
:::

(sec-yukawa-hierarchy-optimization)=
### Conditional Yukawa parameterization and optimization

:::{prf:theorem} Necessary conditions for a supplied spectral optimization model
:label: thm-yukawa-optimal

Suppose a field model specifies
$y_f=Y_0e^{-\Delta\Phi_f/\Phi_0}$ with $Y_0,\Phi_0>0$, and a differentiable
objective $\Lambda(y)$ representing an identified spectral quantity.
At an unconstrained interior extremum,

$$
\frac{\partial\Lambda}{\partial\Delta\Phi_f}
 =-\frac{y_f}{\Phi_0}\frac{\partial\Lambda}{\partial y_f}=0.
$$

The Hessian and any constraints determine whether such a stationary point
is a maximum. Neither existence nor a hierarchy follows from this necessary
condition. For example, the proposed leading model
$\Lambda(y)=\Lambda_0+\sum_fc_fy_f^2$ with $c_f>0$ has no finite interior
stationary point with all $y_f>0$. On $0<y_f\le Y_0$ its maximum occurs at
$y_f=Y_0$ for every $f$.

A Fractal Gas application must derive the Yukawa coefficients, the
identified operator's spectral dependence, and the optimization principle.
A proved convergence-rate bound is not automatically a physical mass or
a differentiable exact spectral gap.
:::

:::{prf:proof}
Differentiate the specified exponential and apply the chain rule.
For the quadratic example,
$\partial\Lambda/\partial\Delta\Phi_f=-2c_fy_f^2/\Phi_0<0$ at every
finite gap, and $\Lambda$ increases in each $y_f$. This proves both the
interior and boundary assertions. $\square$
:::

:::{prf:corollary} Ratios under a common exponential Yukawa ansatz
:label: cor-fermion-mass-ratios

For diagonal positive Yukawa values
$y_f=Y_0e^{-\Delta\Phi_f/\Phi_0}$ with a common $Y_0$ and common Higgs
normalization $m_f=vy_f/\sqrt2$,

$$
\frac{m_f}{m_{f'}}
 =\exp[-(\Delta\Phi_f-\Delta\Phi_{f'})/\Phi_0].
$$

Consequently an observed ratio can determine a gap difference within this
ansatz. Predicting the ratio requires an independent determination of those
gaps. Non-diagonal Yukawa matrices require their singular values rather than
entrywise exponentiation of a mass formula.
:::

:::{prf:proof}
Divide the two specified mass formulas; their common prefactors cancel.
For matrices, biunitary diagonalization defines the positive masses as
singular values, so a formula for individual entries does not substitute
for diagonalization. $\square$
:::

(sec-mixing-angles-optimization)=
### What the proposed mixing objective actually minimizes

:::{prf:proposition} Minimizers of the nonnegative mixing functional
:label: prop-ckm-angles-spectral

For supplied masses $m_i$ define on $U(n_g)$

$$
\mathcal F_{\mathrm{mix}}(V)
 =\sum_{i\ne j}|V_{ij}|^2|m_i^2-m_j^2|.
$$

Its minimum is zero. If the squared masses are pairwise distinct, the
minimizers are precisely the diagonal unitary matrices. With degeneracies,
minimizers can mix only inside equal-mass blocks. Thus unitarity alone in
this objective gives zero inter-mass mixing, not angles proportional to
$\sqrt{m_i/m_j}$.

Additional nonzero-CP or transition constraints can change the problem,
but they must be specified and solved. A separate two-state Hermitian mass
ansatz $\left(\begin{smallmatrix}a&b\\\bar b&c\end{smallmatrix}\right)$
gives, after rephasing $b$ to be real,
$\tan(2\theta)=2|b|/(c-a)$ when $c\ne a$.
A square-root mass-ratio relation would require an additional condition on
$b$ and the diagonal entries.
:::

:::{prf:proof}
Every summand is nonnegative and a diagonal unitary makes them all zero.
For distinct squared masses, zero energy forces all off-diagonal entries
to vanish. If masses are degenerate, it only forces entries between
unequal-mass blocks to vanish; unitarity allows arbitrary unitary matrices
inside each block. For the two-state formula, use a diagonal rephasing to
make $b\ge0$ and a real rotation. Setting the rotated off-diagonal entry
to zero gives $b\cos2\theta-(c-a)\sin2\theta/2=0$.
$\square$
:::

(sec-sm-wilson-loops)=
## 8. Loop Observables and a Finite Field Action

:::{div} feynman-prose
A recorded Wilson loop uses the transport matrices assigned to its edges.
For the Fractal Set attribution connection, an interaction triangle compares
the IA and IG transports through its CST edge. Its Wilson defect was
calculated earlier from those matrices. We now follow its probability law,
together with the companion doublets and color contractions, through the
complete algorithmic update.

The algorithm also supplies the probabilities of these readouts. Average its
complete path likelihood over reference runs with the same descriptor
history. The resulting density gives the effective action, while successive
prefix densities give the next-observation probabilities. The same density
generates channel moments and connected correlations. These formulas retain
the companion draws, cloning, kinetic update, and masks in their implemented
order.
:::

:::{prf:definition} Wilson observable with the chosen representation
:label: def-sm-wilson-loop

For an oriented closed loop $\gamma=(i_0,\ldots,i_m=i_0)$ and unitary links
in a representation of rank $r_G$, define

$$
W[\gamma]=\operatorname{Tr}(U_{i_0i_1}\cdots U_{i_{m-1}i_0}),
\qquad w[\gamma]=W[\gamma]/r_G.
$$

Endpoint gauge transformations cancel except for conjugation at $i_0$,
so both traces are gauge invariant and $|w|\le1$. For an abelian connection
on a loop bounding a surface, Stokes' theorem gives its flux expression.
A nonabelian loop requires the ordered transport product.
These are the observables of {prf:ref}`def-wilson-loop-lqft` for the supplied
field representation, independently of the population size $N$.
:::

:::{prf:proposition} Static potential under an area-law hypothesis
:label: prop-sm-area-law

For a specified Euclidean field measure, suppose
$\langle w(R,T)\rangle>0$ and

$$
\log\langle w(R,T)\rangle=-\sigma RT-\mu_0T+o(T)
$$

at fixed $R$ as $T\to\infty$. Then its defined static-potential limit is
$-\lim_{T\to\infty}T^{-1}\log\langle w(R,T)\rangle=\sigma R+\mu_0$.
The area law must be established for that field measure and loop regime.
Short-range viscous interactions or concentration in fitness basins alone
provide no such Wilson-loop estimate.
:::

:::{prf:proof}
Divide the assumed logarithmic asymptotics by $-T$ and take the limit.
This retains any endpoint or perimeter contribution proportional to $T$,
as required by {prf:ref}`prop-area-law`. $\square$
:::

(sec-sm-unified-lagrangian)=
### Specifying the coupled finite model

:::{prf:definition} A finite gauge, matter, Higgs, and Yukawa action
:label: def-sm-total-action

Choose the gauge representations, links, vertex masses, and edge
quadrature already specified. A finite model can use

$$
S_{\mathrm{total}}=S_W+S_{\mathrm f}+S_H+S_Y+S_M,
\qquad
S_W=\sum_{G,P}\beta_{G,P}
 \left(1-\frac1{r_G}\operatorname{ReTr}U_G[P]\right),\quad\beta_{G,P}\ge0.
$$

The actions $S_{\mathrm f}$ and $S_H$ are those defined above. With the
chosen Standard Model representations, conventional right-handed field
notation allows the invariant Yukawa contractions

$$
S_Y=\sum_i\mu_i\left[
\bar Q_{L,i}\widetilde H_iY_u u_{R,i}
+\bar Q_{L,i}H_iY_d d_{R,i}
+\bar L_iH_iY_e e_{R,i}
+\bar L_i\widetilde H_iY_\nu\nu_{R,i}
+\text{conjugate terms}\right],
\quad \widetilde H=i\sigma^2\overline H.
$$

The matrices $Y_f$ act on the separately specified generation space.
A neutral Majorana term is included only under
{prf:ref}`thm-sm-majorana-mass`. The Higgs potential has $\lambda>0$.
Use normalized Haar integration on independent compact-group links,
Lebesgue integration for the finite scalar variables, and the fixed Berezin
orientation for fermions to define

$$
Z=\int dU\,dH\,[d\bar\psi\,d\psi]\ e^{-S_{\mathrm{total}}}.
$$

After Grassmann integration the scalar integrand is a polynomial times a
confining exponential, so the finite integral is absolutely convergent.
The fermion polynomial can be complex and $Z$ may vanish; a positive
probability interpretation requires more information.

This is a supplied finite field model on the recorded complex. Equality of
its correlators with Fractal Gas sampling requires an identity of measures
or an appropriate weighted estimator. Its continuum application retains
{prf:ref}`cor-continuum-consistency-conditional`, the scalar estimates above,
and the additional spinor, gauge-holonomy, and field-measure requirements.
:::

:::{prf:proof}
$SU(2)$ invariance of $\varepsilon=i\sigma^2$ gives
$\varepsilon\overline B=B\varepsilon$ for $B\in SU(2)$, so
$\widetilde H$ is a doublet with hypercharge $-1/2$.
The four hypercharge sums are respectively
$-1/6-1/2+2/3=0$,
$-1/6+1/2-1/3=0$,
$1/2+1/2-1=0$, and $1/2-1/2+0=0$.
Weak and color indices are contracted between conjugate representations;
the generation matrices commute with the gauge action. This verifies the
Yukawa invariance. The other finite invariances were proved above.

With finitely many Grassmann generators the exponential has only finitely
many terms contributing to its integral. Its coefficients are polynomials
in $H$ with uniformly bounded link coefficients on the compact gauge group.
The positive quartic part of the Higgs potential dominates every such
polynomial at infinity. Positive vertex masses and nonnegative kinetic
terms therefore give an integrable scalar majorant. No positivity of the
resulting fermion polynomial follows from this convergence argument.
$\square$
:::

:::{prf:theorem} Descriptor density from the complete path likelihood
:label: thm-sm-path-descriptor-density

On a standard Borel finite-path space, take the normalized reference path law
$R$ of {prf:ref}`thm-action-from-path-integral` and the actual path law
$P=\mathcal L R$, where $\mathcal L\ge0$, $\int\mathcal L\,dR=1$.
Thus $\mathcal L=e^{-\mathcal S_h}$ with the complete likelihood, including
its initial factor, when written in that theorem's notation. For a measurable
descriptor map $\mathscr D$ into a standard Borel space, set

$$
\lambda=\mathscr D_*R,\qquad \nu=\mathscr D_*P.
$$

Then $\nu=a\lambda$, where a version of its density is

$$
a(y)=\mathbb E_R[\mathcal L\mid\mathscr D=y].
$$

In particular the effective descriptor action is $-\log a$ where $a>0$,
with value $+\infty$ where $a=0$. It is the conditional integral of the
likelihood, followed by the logarithm.

*Proof.* For bounded measurable $f$, the defining conditional-expectation
identity gives

$$
\begin{aligned}
\int f\,d\nu
&=\int f(\mathscr D(s))\mathcal L(s)\,dR(s)\\
&=\int f(\mathscr D(s))
       \mathbb E_R[\mathcal L\mid\sigma(\mathscr D)](s)\,dR(s)\\
&=\int f(y)a(y)\,d\lambda(y).
\end{aligned}
$$

This proves the density formula and $\int a\,d\lambda=1$. The underlying
likelihood is the already proved complete-kernel likelihood, so companion,
cloning, and kinetic contributions retain their original references. For
survival conditioning on an event $E$ of positive $P$ probability, replace
$\mathcal L$ by $\mathcal L\mathbf1_E/P(E)$ before taking the conditional
expectation. $\square$
:::



:::{div} feynman-prose
Fix an observed channel history. Many complete runs can produce it, with
different intermediate arrays and random choices. Conditional averaging of
the complete likelihood assigns their combined weight to that history.
Taking its negative logarithm gives the effective action relative to the
specified reference law.

For the next observation, first condition the current complete state on
the history already seen. Then average the next implemented update over
those states and its fresh random inputs. This gives both the predictive
kernel and the moments of the channel increments below. The history enters
through that conditional state distribution, preserving the memory carried
by the selected channels. Companion selection, cloning, and kinetics all
remain inside the same update calculation.
:::

:::{prf:theorem} Effective gauge-channel action and predictive kernel of the recorded algorithm
:label: thm-sm-effective-recorded-gauge-dynamics

Use the complete update and its random-input law in
{prf:ref}`thm-sm-instantiated-record-transition`. Choose the descriptor
$D_n$ to retain the direct contractions, projector triangles, companion
doublets, and their recorded validity masks at observation $n$; any fixed
subcollection gives the corresponding marginal construction. Transition
descriptors are evaluated on the completed transition record, so their
dependence on companions and intermediate stages is retained. Write
$Y_{0:n}=(D_0,\ldots,D_n)$. Use the consistent reference path law and
likelihood $L_n$ of {prf:ref}`thm-action-from-path-integral`, including the
initial density. Define

(eq-fg-sm-g1)=
$$
\lambda_n=(Y_{0:n})_*R_n,\qquad
a_n(y_{0:n})=\mathbb E_{R_n}[L_n\mid Y_{0:n}=y_{0:n}],\qquad
\nu_n=a_n\lambda_n.
\tag{SM.G1}
$$

Let $k_n^R(y_{0:n},dy)$ be the conditional distribution of $D_{n+1}$
under the reference descriptor law. The actual predictive kernel and
effective path action are, on $a_n>0$,

(eq-fg-sm-g2)=
$$
\begin{aligned}
k_n^P(y_{0:n},dy)
 &=\frac{a_{n+1}(y_{0:n},y)}{a_n(y_{0:n})}
                  k_n^R(y_{0:n},dy),\\
S_n^{\mathrm{eff}}&=-\log a_n,\\
S_{n+1}^{\mathrm{eff}}-S_n^{\mathrm{eff}}
 &=-\log\frac{a_{n+1}}{a_n}.
\end{aligned}
\tag{SM.G2}
$$

Thus the effective action and transition kernel of these gauge channels
are fixed by the actual algorithm. The increment in
{ref}`(SM.G2) <eq-fg-sm-g2>` is a function of the observed history;
its construction does not discard the memory calculated in
{prf:ref}`thm-sm-direct-channel-memory`.

More explicitly, let $\eta_n(ds\mid y_{0:n})$ be the posterior of the
complete Markov-boundary state. For a next-observation map
$d_{n+1}(s,\xi)$ evaluated on the actual completed update, prediction is

(eq-fg-sm-g3)=
$$
\begin{aligned}
\eta_n(B\mid y_{0:n})
 &=\frac{\mathbb E_{R_n}
       [L_n\mathbf1_{\{S_n\in B\}}\mid Y_{0:n}=y_{0:n}]}{a_n(y_{0:n})},\\
\mathbb E_P[f(D_{n+1})\mid Y_{0:n}=y_{0:n}]
 &=\int\eta_n(ds\mid y_{0:n})
              \int f(d_{n+1}(s,\xi))\,m(d\xi).
\end{aligned}
\tag{SM.G3}
$$

The random input contains the companion, cloning, and kinetic choices
in their implemented order. For bounded real channel coordinates $q^a$,
their conditional one-step drift and covariance are consequently

(eq-fg-sm-g4)=
$$
\begin{aligned}
\delta q^a(s,\xi;y_n)&=q^a(d_{n+1}(s,\xi))-q^a(y_n),\\
b_n^a(y_{0:n})&=\int\eta_n(ds\mid y_{0:n})
                         \int\delta q^a\,m(d\xi),\\
C_n^{ab}(y_{0:n})
 &=\int\eta_n(ds\mid y_{0:n})
                         \int\delta q^a\delta q^b\,m(d\xi)
                          -b_n^ab_n^b.
\end{aligned}
\tag{SM.G4}
$$

These are increments per recorded step. Dividing the first two raw
increment moments by the recorded step length gives the corresponding
scaled increment quantities; a diffusion limit is not used in these
finite-step identities.

*Proof.* The density statement is
{prf:ref}`thm-sm-path-descriptor-density` applied to each prefix of the
same path. Consistency of both path laws implies, for every bounded
test $g$ on prefixes,

$$
\begin{aligned}
\int g\,a_n\,d\lambda_n
 &=\int g(y_{0:n})a_{n+1}(y_{0:n},y)
                       k_n^R(y_{0:n},dy)\lambda_n(dy_{0:n}).
\end{aligned}
$$

Uniqueness of densities gives
$\int a_{n+1}(y_{0:n},y)k_n^R(y_{0:n},dy)=a_n(y_{0:n})$.
The ratio in {ref}`(SM.G2) <eq-fg-sm-g2>` is therefore normalized.
Testing it against a bounded function of both prefix and next observation
proves that it is the actual conditional law. Prefixes with $a_n=0$
have zero actual probability and require no transition identification.
Taking logarithms on the positive-density set proves the action formula;
zero conditional density gives infinite action.

Bayes' identity, tested against bounded prefix functions, gives the
first line of {ref}`(SM.G3) <eq-fg-sm-g3>`. Given the complete current
state, the next fresh input has law $m$, and the recorded update is the
map already proved in {ref}`(SM.K1) <eq-fg-sm-k1>`. Conditioning first
on this state and then on the descriptor history proves the second line.
Apply it to each increment and product of two increments to obtain
{ref}`(SM.G4) <eq-fg-sm-g4>`. In particular
$u_aC_n^{ab}u_b=\operatorname{Var}(\sum_a u_a\delta q^a\mid Y_{0:n})\ge0$.
Bounded direct contractions and bounded masked averages make these
integrals finite without an additional moment condition.

For the conservative update $P_h=C_hK_h$, the inner integral is
the composition of the actual cloning and kinetic kernels. Deterministic
substeps and atomic outcomes stay inside these kernels. They are not
replaced by Gaussian densities. For a separately specified conditioned
finite path, the same argument uses its likelihood
$L_n^E=\mathbb E_R[L_K\mathbf1_E\mid\mathcal F_n]/P(E)$ at prefix $n$.
The future survival weight is then already present in the prefix
likelihood; it is not replaced by an unconditioned fresh-input law in
{ref}`(SM.G3) <eq-fg-sm-g3>`. This agrees with the chapter's distinction
between conservative, killed, and Doob transitions. $\square$
:::

:::{div} feynman-prose
The same recorded law can collect all joint moments in one expression.
Multiply each history's weight by the exponential of a source times each
chosen readout. Differentiating with respect to a source brings down that
readout; several derivatives bring down their product. Differentiating the
logarithm gives connected quantities, including the covariance. At zero
source the weights are exactly those of the algorithm, with the original
validity masks. This provides a common calculation for the moments of the
selected gauge composites.
:::

:::{prf:corollary} Exact generating functional for the recorded gauge composites
:label: cor-sm-recorded-gauge-generating-functional

In {prf:ref}`thm-sm-effective-recorded-gauge-dynamics`, let
$O_1,\ldots,O_r$ be bounded real direct channel observables on a fixed
finite record. In particular one may use the real and imaginary parts
of projector triangles, with the recorded zero convention at invalid
vertices. Their source functional is

(eq-fg-sm-g5)=
$$
Z(J)=\mathbb E_R\left[L_n
                 e^{\sum_{\alpha=1}^rJ_\alpha O_\alpha}\right]
     =\int e^{J\cdot O(y)}e^{-S_n^{\mathrm{eff}}(y)}\lambda_n(dy),
\qquad Z(0)=1.
\tag{SM.G5}
$$

It is entire in $J\in\mathbb C^r$. Its derivatives at zero are every
joint moment of this selected collection. For real $J$, put $W(J)=\log Z(J)$.
Then

(eq-fg-sm-g6)=
$$
\partial_\alpha W(J)=\mathbb E_{P_J}O_\alpha,
\qquad
\partial_\alpha\partial_\beta W(J)
 =\operatorname{Cov}_{P_J}(O_\alpha,O_\beta),\qquad
dP_J=Z(J)^{-1}e^{J\cdot O}dP.
\tag{SM.G6}
$$

Here the source is a tool for extracting the actual correlations;
at $J=0$ the law is exactly the recorded algorithmic law.

*Proof.* Since $|O_\alpha|\le M_\alpha$, every derivative of the
integrand on a compact source set is bounded by a constant times $L_n$.
The exponential power series is absolutely dominated by
$L_n\exp(\sum_\alpha |J_\alpha|M_\alpha)$, whose integral is finite.
Termwise integration proves entire dependence and the moment formula.
For real sources $Z>0$. Direct differentiation of $\log Z$ gives

$$
\frac{\partial_\alpha\partial_\beta Z}{Z}
 -\frac{\partial_\alpha Z\,\partial_\beta Z}{Z^2}
=\mathbb E_{P_J}(O_\alpha O_\beta)
 -\mathbb E_{P_J}O_\alpha\mathbb E_{P_J}O_\beta.
$$

The density identity proves the other expression in
{ref}`(SM.G5) <eq-fg-sm-g5>`. Thus the effective action, the predictive
kernel, and the channel generating functional are three expressions
for one pushforward law. $\square$
:::

:::{div} feynman-prose
The color formula stores force direction in its component magnitudes and
velocity in its phases. Compare nearby recorded colors through their
normalized overlaps: the accumulated phase around a small closed path
measures the curvature computed below.

The explicit example varies actual swarm velocities while keeping positions
fixed. The full viscous force sum then produces a rotating force of fixed
nonzero magnitude, so every sample on the chosen surface passes the force
mask. Varying the force angle and the common velocity gives a nonzero
curvature on this surface of swarm states. The finite overlap triangles
and their history law provide the corresponding recorded observables.
:::

:::{prf:proposition} Curvature of the recorded color phase quotient
:label: prop-sm-recorded-color-connection-curvature

On the finite-dimensional valid descriptor domain $F\ne0$, write the
recorded color formula as
$c^a=r_a e^{i\kappa v^a}$, where
$r_a=F_a/\|F\|$ and $\kappa=m\ell_0/\hbar_{\mathrm{eff}}$.
The phase quotient in {prf:ref}`thm-sm-direct-phase-quotient` has the
local connection one-form and curvature

(eq-fg-sm-g7)=
$$
\begin{aligned}
\mathcal A&=-i c^\dagger dc
                 =\kappa\sum_a\frac{F_a^2}{\|F\|^2}\,dv^a,\\
\mathcal F&=d\mathcal A
 =\frac{2\kappa}{\|F\|^2}\sum_a
 \left(F_a\,dF_a-\frac{F_a^2}{\|F\|^2}
                           \sum_bF_b\,dF_b\right)\wedge dv^a.
\end{aligned}
\tag{SM.G7}
$$

Under a local representative change $c'=e^{i\alpha}c$,
$\mathcal A'=\mathcal A+d\alpha$ and $\mathcal F'=\mathcal F$.
The normalized overlap links already present in
{prf:ref}`prop-sm-direct-triangle-projectors` have this connection
as their infinitesimal comparison. It is the $U(1)$ connection of
the recorded color line, not a new independent $SU(3)$ link variable.

*Proof.* Since $\sum_ar_a^2=1$, differentiation gives
$\sum_ar_a\,dr_a=0$. Therefore

$$
c^\dagger dc
 =\sum_a r_a\,dr_a+i\kappa\sum_a r_a^2\,dv^a
 =i\kappa\sum_a r_a^2\,dv^a.
$$

Furthermore

$$
d\left(\frac{F_a^2}{\|F\|^2}\right)
=\frac{2F_a\,dF_a}{\|F\|^2}
 -\frac{2F_a^2\sum_bF_b\,dF_b}{\|F\|^4},
$$

which proves {ref}`(SM.G7) <eq-fg-sm-g7>`. Substitution of
$c'=e^{i\alpha}c$ gives $-ic'^\dagger dc'=d\alpha-ic^\dagger dc$;
exterior differentiation proves curvature invariance.

For a smooth parameter curve $s(t)=(F(t),v(t))$ in this descriptor domain,
$c(s(t))^\dagger c(s(t+h))=1+ih\mathcal A_{s(t)}(\dot s(t))+O(h^2)$.
Its modulus is $1+O(h^2)$, so normalization preserves the linear term.
Products of the normalized overlaps around a partitioned smooth loop
therefore converge to $\exp(i\oint\mathcal A)$. On a loop bounding a
surface inside a representative chart this equals
$\exp(i\int\mathcal F)$ by Stokes' formula. These are parameter-space
identities for the recorded descriptor map.

The nonzero curvature can be realized by the actual same-stage viscous
feature map, rather than by varying $F$ independently of the swarm.
Use the active viscous sector of
{prf:ref}`cor-ym-nonzero-direct-color-sector`. Fix regular positions and
an interacting pair $i,j$ with $\nu K_{ij}\ne0$, retaining all their
position-dependent scalar weights. In the force convention of
{prf:ref}`thm-sm-su3-emergence`, select a fixed $R>\delta_c$ and set

$$
v_i=u e_1,\qquad
v_k=v_i\quad(k\ne j),\qquad
v_j=v_i+\frac{R}{\nu K_{ij}}
                  (\cos\theta\,e_1+\sin\theta\,e_2).
$$

Every summand of the actual force except $j$ vanishes, including any
self-weight term. Thus the complete force sum on this state surface is

$$
F_i^{\mathrm{visc}}
 =\nu\sum_kK_{ik}(v_k-v_i)
 =R(\cos\theta\,e_1+\sin\theta\,e_2),\qquad
\|F_i^{\mathrm{visc}}\|=R>\delta_c.
$$

The mask is one throughout this surface. Substitution into the recorded
normalized phase formula gives
$c_i=(\cos\theta\,e^{i\kappa u},\sin\theta,0)$, and hence

(eq-fg-sm-g8)=
$$
\mathcal A=\kappa\cos^2\theta\,du,\qquad
\mathcal F=-2\kappa\sin\theta\cos\theta\,d\theta\wedge du.
\tag{SM.G8}
$$

For $\kappa\ne0$ and $0<\theta<\pi/2$ this is a nonzero curvature
of the actual feature map on a finite-dimensional swarm-state surface.
The fixed nonzero interaction weight makes that surface smooth in
$(\theta,u)$; regular position neighborhoods preserve the nonzero-weight
condition. Positive density of the established product reference, and
the lower density comparison for its bounded-tilt family, give positive
probability to open neighborhoods of these regular states. The
two-dimensional surface itself need not have positive probability.
This assertion concerns those identified laws; a separately selected
law uses its own support identification.

The finite triangle formula already records the corresponding
nontrivial overlap phases without taking a limit. For actual constrained
records, the forms in {ref}`(SM.G7) <eq-fg-sm-g7>` are pulled back along
their existing feature map, as the explicit force-sum substitution above
demonstrates. This state-space curvature is not an identification with
spacetime Yang--Mills curvature. Its actual loop distribution and
time evolution are precisely the pushforward law and prediction formulas
{ref}`(SM.G1) <eq-fg-sm-g1>`--{ref}`(SM.G4) <eq-fg-sm-g4>`.
$\square$
:::


:::{prf:proposition} Equality of field measures and weighted estimators
:label: prop-sm-field-measure-comparison

Use the same descriptor space and base measure $\lambda$ as in
{prf:ref}`thm-sm-path-descriptor-density`. Suppose the integrated finite field
model is represented there by an integrable density $b$, with
$Z_b=\int b\,d\lambda\ne0$. Its normalized functional is
$\langle f\rangle_b=Z_b^{-1}\int fb\,d\lambda$.
Equality with the direct law for every bounded measurable $f$ holds precisely
when $b/Z_b=a$ almost everywhere.

An integrable weighting of the direct samples represents this functional
precisely when $b=0$ almost everywhere on $\{a=0\}$. In that case define
$w=b/a$ on $\{a>0\}$ and zero elsewhere. For $\int|fb|d\lambda<\infty$,

$$
\langle f\rangle_b=\frac{\mathbb E_\nu[wf]}{\mathbb E_\nu[w]},
\qquad \mathbb E_\nu[w]=Z_b,\qquad
\mathbb E_\nu|w|=\int|b|\,d\lambda.
$$

For a finite-variance estimator one also checks, for example,
$\mathbb E_\nu|wf|^2=\int_{a>0}|bf|^2/a\,d\lambda<\infty$.
These identities concern expectations; a ratio of finite empirical means
has its own sampling error. A complex fermion density produces complex weights
and a normalized functional, rather than a positive probability law.

*Proof.* Equality of all bounded integrals is equality of the two finite
measures, hence equality of their densities. For weighting, multiplication
by $a$ gives $wa=b$ on $\{a>0\}$ and gives zero on $\{a=0\}$. Thus the
stated zero-set condition is necessary and sufficient. Integrating the
identities $wa=b$, $|w|a=|b|$, and $|wf|^2a=|bf|^2/a$ proves each formula.
If an action is originally defined against a different reference, its
Radon--Nikodym factor must be included in $b$ on this common space.
$\square$
:::



:::{prf:remark} Dependency order for the represented field theory
:label: rem-sm-proof-dependency-order

The finite reconstruction identity is proved first, using the codec and
recorded-sample coverage. Gram and determinant algebra then proves orbit
separation and the explicit inverse. Pushforward integration proves the
observable-space unitary; it uses no Standard Model action or gauge-law
invariance. The static LSI and sampling estimates enter from their own
Volume 2 proofs through the actual pullback and the channel calculations in
{prf:ref}`prop-sm-channel-estimate-routes`.

For the fermionic route, the order is exterior inner product, creation and
contraction, CAR, determinant integration on independent replicas, and finally
transition intertwining. The two replica/Fock theorems cross-reference the
same explicit determinant calculation; their conclusions need not be used
as premises of one another. The generator is then
{prf:ref}`prop-sm-replica-generator`.

The complete-kernel likelihood yields the descriptor density and the
effective action in {prf:ref}`thm-sm-effective-recorded-gauge-dynamics`.
Its successive prefix densities give the predictive kernel; source
derivatives give all channel correlations. The interaction transports
enter this same descriptor law through
{prf:ref}`prop-sm-attribution-holonomy-defect`. The full recorded
kernel also constructs the prediction-complete channel representation
and its convergent transition matrices. A separately supplied field action
is compared with this algorithm-derived law using
{prf:ref}`prop-sm-field-measure-comparison` and
{prf:ref}`prop-sm-field-generator-comparison`.
:::

(sec-sm-dictionary)=
### The proved correspondence and its remaining identifications

:::{prf:definition} Mathematical correspondence table
:label: def-sm-dictionary

The following table records the objects and the conclusions established
in this chapter. A representation assignment is part of the supplied field
model; a recorded identity is a statement about the actual finite data.

| Object | Established statement | Additional identification for physics |
|---|---|---|
| Full color Gram matrix and complex triple determinants | Homeomorphism with the SU(3) orbit space and an explicit anchor-chart inverse | Identify any selected physical spectral sector |
| Full doublet Hermitian and alternating contractions | Homeomorphism with the SU(2) orbit space, without Dirac matrices | Identify weak dynamics in the same law |
| Invariant history coordinates | Unitary representation of all integrable correlations; prediction-complete Markov extension and convergent finite transition matrices | Estimate the matrix entries under the actual recorded law |
| Direct descriptor law | Exact prefix action, full-history predictive kernel, and generating functional from the complete likelihood | Evaluate the descriptor integrals in the chosen algorithm regime |
| Color triangle product | Trace of three rank-one projectors; exact independent-phase invariance | Distinguish this composite from a unitary-link Wilson observable |
| Companion amplitudes | Normalization and phase freedom | A nontrivial connection and field measure |
| Cloning doublet and Fractal Set attribution connection | SU(2) invariant doublet algebra; interaction holonomy and exact IA/IG Wilson mismatch | Evaluate the transports and their correlations through the complete recorded update |
| Viscous force | Orthogonal covariance and exact force moments | Covariant internal color field |
| Score antisymmetry and record modes | Weighted sign identity; exact CAR and antisymmetric-replica realization for centered record modes | Equality with the chosen interacting field generator and measure |
| Exterior spinors | Faithful complex Clifford representation | Spin geometry and a Dirac operator limit |
| Quartic potential | Minima, Hessian, and chosen Higgs mass matrix | Fractal Gas reduction and field normalization |
| Spin(10) half-spin field | Sixteen-state branching, vanishing local anomaly coefficients, and even weak-doublet count | A map from recorded states and its dynamics |
| Generation space | Arbitrary representation multiplicity | A principle determining $n_g$ |
| Role observables | Exact partition and zero same-frame delta-to-right statistic | A physical spectral channel for another measured statistic |
| Coupling proxies | Defined moments, units, and bounds | Canonical coupling matching |
| CP quartet | Rephasing invariant, odd under conjugation | An identified ensemble with nonzero expectation |
| Neutral mass matrix | Majorana gauge criterion and seesaw estimate | An actual generated coefficient and continuum localization |
| Yukawa ansatz | Conditional ratios and necessary optimization conditions | Derived gaps and a justified optimization principle |
| Wilson loops | Gauge invariance and conditional static potential | An area law in the specified field measure |
:::

:::{div} feynman-prose
For the direct construction, we can follow an observable all the way through:
reconstruct its inputs from the Fractal Set, apply the implemented update in
complete encoded coordinates, and evaluate the readout. Retaining the required
state and transition data makes this construction exact for every integrable
finite history observable, including masked measurements. The exterior
isomorphism also supplies the CAR replica representation and its evolution.

The established regularity and LSI results supply estimates through the
specified pullbacks. The equilibrium energy construction uses its identified
law to derive another generator, with evolution time $\sigma$. Associating
each estimator with its law and clock tells us which of these estimates to
use. The path likelihood and generator comparisons specify how to test a
proposed coupled field model against the constructed evolution. The charge
and anomaly calculations establish the representation properties of the
fields used in that model.
:::
