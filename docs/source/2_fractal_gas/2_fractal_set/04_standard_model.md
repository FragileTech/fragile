# Direct Observables and Standard Model Representations on the Fractal Set

(sec-sm-introduction)=
## 1. Recorded Data and the Direct Observable Formulation

:::{div} feynman-prose
Start with the quantities the gas records: forces, velocities, fitness, and
companions. The Fractal Set reconstruction supplies these inputs; explicit
formulas turn them into complex vectors and their contractions. The complete
Gram and determinant coordinates then recover the descriptor configuration
up to a common special-unitary frame change. Carrying the recorded law
through this map preserves every represented correlation exactly.

The volume supplies more than coordinates. Its derivative bounds, ellipticity,
and LSI estimates control the specified fields and laws, while the exterior
construction identifies CAR operators with antisymmetric replicas of the
whole swarm. We use these results through their explicit maps. Comparing the
resulting dynamics with a coupled field action requires the corresponding
measure and generator calculations. Keeping those maps visible tells the
simulator precisely which observable to compute and which law to sample.
Dirac matrices are an optional representation of the operator algebra.
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

The doublet frames also give genuine SU(2) transport matrices. Follow what
this particular transport does: it carries the normalized doublet at one end
of an edge exactly onto the doublet at the other end. Their covariant difference
vanishes, and going around a closed loop returns the identity. Allow the
doublet lengths to vary and the edge energy measures only that change in
length. These are exact consequences of the frame formula, useful when
choosing which link variables a simulator must retain.
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
allowed-companion set and normalization. Assigning node phases
$\theta_i=-\Phi_i/\hbar_{\mathrm{eff}}$ and
$U_{ij}=e^{i(\theta_i-\theta_j)}$ gives trivial holonomy on every loop.

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
$e^{i\alpha_i}$. Products of node-difference links telescope around loops.
Substitution of $V_i+b$ into the score gives the displayed positive row
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
these invariant coordinates retains its value.

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

:::{prf:proposition} Frame links constructed from normalized doublets
:label: prop-sm-direct-su2-frames

For $z\in\mathbb C^2$ with $\|z\|=1$, define

$$
B(z)=\begin{pmatrix}z_1&-\overline{z_2}\\z_2&\overline{z_1}\end{pmatrix}.
$$

Then $B(z)\in SU(2)$ and $B(\Omega z)=\Omega B(z)$ for
$\Omega\in SU(2)$. For a chosen graph on doublet indices,
$U_{ij}=B(z_i)B(z_j)^\dagger$ has inverse reversal and transforms as
$U_{ij}\mapsto\Omega_iU_{ij}\Omega_j^{-1}$. Every closed product of these
links equals $I$. They realize a flat comparison convention from vertex
frames. Nontrivial local $SU(2)$ holonomy requires a further transport
construction, with its covariance and law established separately.
:::

:::{prf:proof}
The second column is $Jz=(-\overline z_2,\overline z_1)^{\mathsf T}$.
It satisfies

$$
z^\dagger Jz=-\overline z_1\overline z_2+
\overline z_2\overline z_1=0,\qquad
\|Jz\|^2=\|z\|^2=1,
\qquad \det[z,Jz]=|z_1|^2+|z_2|^2=1.
$$

Thus $B(z)^\dagger B(z)=I$ and $\det B(z)=1$. Any unitary matrix
with first column $z$ has second column $e^{i\beta}Jz$, since the
orthogonal complement is one-dimensional. Its determinant is $e^{i\beta}$;
requiring determinant one forces that column to equal $Jz$. This proves
uniqueness of the completion. Both $B(\Omega z)$ and $\Omega B(z)$
belong to $SU(2)$ and have first column $\Omega z$, so they coincide.

Write $B_i=B(z_i)$. Then

$$
U_{ij}^\dagger=(B_iB_j^\dagger)^\dagger=B_jB_i^\dagger=U_{ji},
\qquad U_{ij}U_{ji}=B_i(B_j^\dagger B_j)B_i^\dagger=I.
$$

Equivariance gives
$U'_{ij}=(\Omega_iB_i)(\Omega_jB_j)^\dagger
=\Omega_iU_{ij}\Omega_j^{-1}$.
For a closed sequence $i_0,\ldots,i_r=i_0$,

$$
\prod_{k=0}^{r-1}U_{i_ki_{k+1}}
=B_{i_0}(B_{i_1}^\dagger B_{i_1})\cdots
 (B_{i_{r-1}}^\dagger B_{i_{r-1}})B_{i_0}^\dagger=I.
$$

This establishes inverse reversal, covariance, and the exact flatness of
this particular frame construction. $\square$
:::

:::{prf:corollary} Covariant differences for the doublet-frame links
:label: cor-sm-frame-link-radial-action

For the links of {prf:ref}`prop-sm-direct-su2-frames`,

$$
U_{ij}z_j=z_i,\qquad U_{ij}z_j-z_i=0.
$$

For radii $r_i\ge0$ and fields $H_i=r_i z_i$, the spatial term of
{prf:ref}`def-sm-scalar-action` reduces to

$$
\frac12\sum_{\{i,j\}}c_{ij}\|U_{ij}H_j-H_i\|^2
=\frac12\sum_{\{i,j\}}c_{ij}(r_j-r_i)^2.
$$

Every weak Wilson-face term built from these links is zero. Thus this
particular matrix construction supplies exact frame comparisons and radial
kinetics. Its link variables have no independent curvature fluctuations.

*Proof.* The first column identity is $B(z_j)e_1=z_j$. Unitarity gives
$B(z_j)^\dagger z_j=e_1$, so
$U_{ij}z_j=B(z_i)e_1=z_i$. Consequently
$U_{ij}H_j-H_i=(r_j-r_i)z_i$, whose squared norm is $(r_j-r_i)^2$.
The closed product is $I_2$ by the preceding proposition, and
$1-\tfrac12\operatorname{ReTr}I_2=0$. These statements hold for every
record before any averaging or limit. $\square$
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
The volume already constructs CAR operators and an exact isomorphism with
antisymmetric replicas. A one-particle vector in that construction is a
function of the complete swarm state. In the two-particle sector, imagine two
independent copies of the entire swarm, then antisymmetrize their observables.
Each copy retains all the interactions among its own walkers. Its evolution
acts on each replica in turn, which explains the sum of generators appearing
in the proof.

This identifies both an operator algebra and its replica evolution. To compare
it with a specified gauge or Yukawa evolution, we must also intertwine the
generators on the domains stated below. The formulas make that comparison
an operator calculation. Clifford and Dirac matrices provide another
representation of the algebra; the CAR construction itself does not depend
on choosing them.

Keep the products straight when implementing the formulas. Recorded amplitudes
multiply as complex numbers; the exterior product alternates. The determinant
observables give the explicit connection, with their expectations taken in
the specified law.
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

:::{prf:assumption} Exterior-algebra field representation
:label: axm-sm-grassmann

Choose independent generators $\psi_{i,a},\bar\psi_{i,a}$ in a finite
exterior algebra, where $a$ indexes the supplied internal and spinor fields.
Distinct generators anticommute and each squares to zero. The bar is an
independent generator in Euclidean Grassmann integration. This is the finite Grassmann representation of {prf:ref}`post-grassmann`.
For the recorded process, the CAR algebra, vacuum state, replica measure,
and their exact correlation identities have already been constructed in
{prf:ref}`thm-lqft-record-fock-reconstruction`,
{prf:ref}`thm-lqft-replica-isomorphism`, and
{prf:ref}`cor-sm-direct-fock-isomorphism`. A Berezin action with a separately
chosen coefficient matrix uses its own measure; its equality with that
recorded-process representation is checked through
{prf:ref}`prop-sm-field-generator-comparison` and the finite measure comparison
below.

In particular $\bar\psi_j\psi_i=-\psi_i\bar\psi_j$.
Anticommutation does not give
$\bar\psi_j\psi_i=-\bar\psi_i\psi_j$, since those monomials involve
different generators.
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
A Wilson loop is a function of links; its expectation depends on their joint
law. We have an exact path-likelihood formula for the algorithm. To obtain
the likelihood seen by its descriptors, average that path likelihood over
reference paths with the same descriptor data. This is the conditional
expectation calculated below, and it gives the precise density to compare
with a proposed field action.

The comparison includes the configurations each law can reach. Doublet-frame
links close to the identity around every loop. Reweighting those samples
changes their probabilities while preserving that constraint. Independent
Haar links allow nonidentity holonomy, so an ordinary reweighting of the flat
samples cannot sample their full law. The finite-action construction below
specifies its own integration variables and measure, making this comparison
explicit before a loop expectation is interpreted physically.
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

:::{prf:proposition} Flat-link support cannot sample an independent-link model
:label: prop-sm-flat-link-support

Fix a finite graph containing a simple cycle and independent $SU(2)$ link
variables on its unoriented edges, with reverse links defined as inverses.
Let $\mathcal Z$ be the set for which the ordered holonomy of that cycle is
$I_2$. Product Haar measure assigns $\mathcal Z$ measure zero. The pushforward
of the doublet-frame construction assigns it probability one.
Consequently no integrable weighting of only those flat-link samples can
represent a normalized field measure absolutely continuous with respect to
product Haar measure. This also holds for a finite complex field measure
with nonzero total mass and such absolute continuity.

*Proof.* Condition on all links except one edge occurring once in the simple
cycle. Its holonomy is $AUB$ or $AU^{-1}B$ for fixed $A,B\in SU(2)$.
Haar measure and its inverse are invariant under left and right translations,
so this conditional holonomy is Haar distributed. A singleton has Haar measure
zero: if it had mass $c>0$, translation invariance would give any $m$ distinct
points mass $mc$, contradicting normalization for large $m$.
Integrating the conditional probability proves the first assertion.
Flatness in {prf:ref}`prop-sm-direct-su2-frames` proves the second.
Every weighted direct measure remains supported on $\mathcal Z$, whereas
absolute continuity of the field measure gives it mass zero there. A nonzero
normalized measure cannot satisfy both properties. $\square$
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

The complete-kernel likelihood yields the descriptor density independently
of the supplied field action. Its comparison with that action uses
{prf:ref}`prop-sm-field-measure-comparison`, including the support test.
Equality of evolutions uses {prf:ref}`prop-sm-field-generator-comparison`.
Only after these comparisons may properties of the field action be transferred
back to the record process, or conversely. The flat-support calculation
specifies why the doublet-frame links alone do not realize the independent-link
integral. Representation, anomaly, and finite-action calculations remain
available without making that identification circularly.
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
| Invariant history coordinates | Unitary observable-space representation preserving all integrable correlations | Apply the established transfer theorem to its own law |
| Direct descriptor law | Conditional-likelihood density and inherited LSI with its induced energy | Channel domain, support-compatible measure comparison, and generator intertwining |
| Color triangle product | Trace of three rank-one projectors; exact independent-phase invariance | Distinguish this composite from a unitary-link Wilson observable |
| Companion amplitudes | Normalization and phase freedom | A nontrivial connection and field measure |
| Cloning doublet and frame links | Exact SU(2) matrices, flat holonomy, and radial-only kinetic term | Connection dynamics satisfying the measure and generator comparisons |
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
reconstruct its inputs from the Fractal Set, compute its invariant coordinates,
and evaluate it under the pushed-forward record law. The full-coordinate
isomorphism preserves the represented information and its correlations.
The established regularity and LSI results supply estimates through the
specified pullbacks; the exterior isomorphism supplies the CAR replica
representation and its evolution.

An implementation can preserve these identities by retaining the descriptor
data and masks before averaging, recording the reconstruction parameters,
and associating each estimator with its law and time coordinate. The path
likelihood and generator comparisons specify how to test a proposed coupled
field model against that evolution. The charge and anomaly calculations
then have a clear role: they establish the representation properties of the
fields used in that model.
:::
