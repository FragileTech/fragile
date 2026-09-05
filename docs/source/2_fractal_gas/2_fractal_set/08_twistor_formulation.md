# Twistor Formulation of the Fractal Set

**Prerequisites**: {doc}`01_fractal_set`, {doc}`03_lattice_qft`, {doc}`04_standard_model`,
{doc}`05_yang_mills_noether`, {doc}`09_qft_calibration`


## TLDR

This chapter has two logically distinct parts.

1. It records the **exact continuum two-twistor formalism** for a reconstructed Lorentzian
   momentum bispinor $P_{AA'}$, including the standard identity

   $$
   m^2 = \det P_{AA'} = \langle \lambda_1 \lambda_2 \rangle
   [\tilde{\lambda}_1 \tilde{\lambda}_2].
   $$

2. It then states the **implemented Fractal Set operator construction**, which is different in
   status. The current code does **not** assign a twistor mass directly to a walker or to a
   companion pair. Instead it builds effective twistor-like local operators on source-frame
   companion triplets and feeds those operators into the existing Euclidean correlator pipeline.

The exact mass identity is therefore background geometry. The implemented observables are

$$
\mathcal{O}_{\mathrm{S}} = \Re \tau,
\qquad
\mathcal{O}_{\mathrm{P}} = \Im \tau,
\qquad
\mathcal{O}_{\mathrm{G}} = |\tau|^2,
$$

where $\tau$ is a twistor-inspired bracket built from the distance and clone companion edges of a
single source walker. These are **operator fields**, not particle masses. Masses are extracted only
from the long-time decay of their correlators, exactly as in {doc}`09_qft_calibration`.


(sec-twistor-intro)=
## Introduction

The preceding chapters establish three facts that make a twistor language useful but also force a
careful separation of levels. First, walkers are the microscopic degrees of freedom of the
simulation, not particles. Second, particles are identified in the analysis layer through poles of
Euclidean correlators, as in {doc}`09_qft_calibration`. Third, the Fractal Set nevertheless stores
enough geometric data on nodes and edges to define local spinor-valued and bispinor-valued
observables ({prf:ref}`def-fractal-set-spinor-space`,
{prf:ref}`def-fractal-set-cst-attributes`, {prf:ref}`def-lqft-chiral-projectors`).

Twistor theory packages null directions and conformal incidence relations in a geometric form
tailored to Lorentzian kinematics. That exact formalism is important background, and we record it
first. But the implemented Fractal Set channels use it in a weaker, operator-theoretic way: they
construct deterministic twistor-inspired local observables from walker/companion geometry and then
measure the spectrum through the same correlator machinery already used for the other channels.

This chapter is therefore deliberately bifurcated.

- Sections 1--4 give the exact continuum twistor identities for a reconstructed Hermitian
  Lorentzian bispinor.
- Sections 5--6 define the effective companion-graph twistor operators that are actually computed
  on the Fractal Set and explain why their correlators, not their pointwise values, are the mass
  observables.


(sec-twistor-spinor-conventions)=
## 1. Spinor Conventions and Momentum Bispinors

Throughout this chapter we work in the physical 3+1-dimensional case of Section 4.7 of
{doc}`03_lattice_qft` with metric signature $(+,-,-,-)$.

:::{prf:definition} Two-Spinor Conventions
:label: def-twistor-spinor-conventions

Let $A,B \in \{0,1\}$ denote undotted spinor indices and let $A',B' \in \{0',1'\}$ denote dotted
spinor indices. The antisymmetric Levi-Civita spinors satisfy

$$
\epsilon_{01} = -\epsilon_{10} = 1,
\qquad
\epsilon^{01} = -\epsilon^{10} = 1.
$$

Index raising and lowering are

$$
\lambda^A = \epsilon^{AB}\lambda_B,
\qquad
\lambda_A = \epsilon_{BA}\lambda^B,
\qquad
\tilde{\lambda}^{A'} = \epsilon^{A'B'}\tilde{\lambda}_{B'},
\qquad
\tilde{\lambda}_{A'} = \epsilon_{B'A'}\tilde{\lambda}^{B'}.
$$

For lower-index representatives we define the invariant brackets

$$
\langle \lambda \chi \rangle := \epsilon^{AB}\lambda_A \chi_B,
\qquad
[\tilde{\lambda} \tilde{\chi}] := \epsilon^{A'B'}\tilde{\lambda}_{A'} \tilde{\chi}_{B'}.
$$

Both brackets are antisymmetric:

$$
\langle \lambda \chi \rangle = - \langle \chi \lambda \rangle,
\qquad
[\tilde{\lambda} \tilde{\chi}] = -[\tilde{\chi} \tilde{\lambda}].
$$
:::

:::{prf:definition} Sigma-Matrix Momentum Map
:label: def-twistor-momentum-map

Let $\sigma_{\mu\,AA'} = (\mathbf{1}, \sigma^1, \sigma^2, \sigma^3)$ with the usual Pauli
matrices. For a Lorentz vector $p^\mu = (p^0, p^1, p^2, p^3)$ define the Hermitian momentum
bispinor

$$
P_{AA'} := p^\mu \sigma_{\mu\,AA'}
=
\begin{pmatrix}
p^0 + p^3 & p^1 - i p^2 \\
p^1 + i p^2 & p^0 - p^3
\end{pmatrix}.
$$
:::

:::{prf:proposition} Determinant Equals the Lorentz Norm
:label: prop-twistor-det-lorentz

For the bispinor of {prf:ref}`def-twistor-momentum-map`,

$$
\det P = p_\mu p^\mu = (p^0)^2 - (p^1)^2 - (p^2)^2 - (p^3)^2.
$$
:::

:::{prf:proof}
Compute directly from the $2 \times 2$ determinant:

$$
\det P
= (p^0 + p^3)(p^0 - p^3) - (p^1 - i p^2)(p^1 + i p^2).
$$

The first product equals $(p^0)^2 - (p^3)^2$. The second equals
$(p^1)^2 + (p^2)^2$. Therefore

$$
\det P
= (p^0)^2 - (p^1)^2 - (p^2)^2 - (p^3)^2
= p_\mu p^\mu,
$$

which is the Lorentz norm in signature $(+,-,-,-)$. $\square$
:::

:::{prf:proposition} Rank-One Factorization Gives a Null Momentum
:label: prop-twistor-null-factorization

If

$$
P_{AA'} = \lambda_A \tilde{\lambda}_{A'},
$$

then $P$ has rank at most $1$ and hence

$$
\det P = 0.
$$

Consequently the associated Lorentz vector is null.
:::

:::{prf:proof}
The matrix $P$ is an outer product of a column and a row. Every outer product has rank at most
$1$, so its determinant vanishes. By {prf:ref}`prop-twistor-det-lorentz`, $\det P = p_\mu p^\mu$,
hence $p_\mu p^\mu = 0$. $\square$
:::


(sec-twistor-space)=
## 2. Twistor Space and the Infinity Twistor

:::{prf:definition} Twistor and Incidence Relation
:label: def-twistor-basic

A **twistor** is a vector

$$
Z^\alpha = (\mu^{A'}, \lambda_A) \in \mathbb{T} \cong \mathbb{C}^4.
$$

At a spacetime point $x^{AA'}$, the incidence relation is

$$
\mu^{A'} = i x^{AA'} \lambda_A.
$$

Thus a fixed point $x$ together with a two-spinor $\lambda_A$ determines a twistor
$Z^\alpha(x,\lambda)$.
:::

The incidence relation shows why a single twistor naturally carries one null direction: the free
data are the two spinor components of $\lambda_A$, and by
{prf:ref}`prop-twistor-null-factorization` the corresponding momentum bispinor is rank one.

:::{prf:definition} Infinity Twistor
:label: def-twistor-infinity

Fix the skew bilinear form $I_{\alpha\beta}$ on $\mathbb{T}$ whose contraction with two twistors
selects the undotted spinor block:

$$
I_{\alpha\beta} Z_1^\alpha Z_2^\beta
:= \epsilon^{AB}\lambda_{1A}\lambda_{2B}
= \langle \lambda_1 \lambda_2 \rangle.
$$

Its conjugate $\bar{I}_{\bar{\alpha}\bar{\beta}}$ selects the dotted block:

$$
\bar{I}_{\bar{\alpha}\bar{\beta}} \bar{Z}_1^{\bar{\alpha}} \bar{Z}_2^{\bar{\beta}}
:= \epsilon^{A'B'}\tilde{\lambda}_{1A'}\tilde{\lambda}_{2B'}
= [\tilde{\lambda}_1 \tilde{\lambda}_2].
$$
:::

The importance of {prf:ref}`def-twistor-infinity` is structural. Twistor space is naturally
conformal, while a scalar mass belongs to Poincare kinematics. The infinity twistor is exactly the
object that chooses the Poincare scale and makes the bracket contractions into physical invariants.


(sec-twistor-massive-system)=
## 3. Massive Two-Twistor Kinematics

:::{prf:definition} Two-Twistor Momentum System
:label: def-twistor-two-system

Let

$$
Z_I^\alpha = (\mu_I^{A'}, \lambda_{IA}),
\qquad
I \in \{1,2\},
$$

be two twistors incident with the same spacetime point $x^{AA'}$:

$$
\mu_I^{A'} = i x^{AA'} \lambda_{IA}.
$$

On the Lorentzian real slice let $\tilde{\lambda}_{IA'}$ denote the conjugate dotted spinors. The
associated momentum bispinor is

$$
P_{AA'} := \lambda_{1A}\tilde{\lambda}_{1A'} + \lambda_{2A}\tilde{\lambda}_{2A'}.
$$
:::

Each summand is rank one and therefore null by {prf:ref}`prop-twistor-null-factorization`. A
timelike momentum arises only when the two null directions are linearly independent.

:::{prf:theorem} Twistor Mass Formula
:label: thm-twistor-mass-formula

For the two-twistor momentum bispinor of {prf:ref}`def-twistor-two-system`,

$$
\det P
= \langle \lambda_1 \lambda_2 \rangle
[\tilde{\lambda}_1 \tilde{\lambda}_2].
$$

Equivalently, the scalar mass of the associated timelike momentum is

$$
m^2
:= p_\mu p^\mu
= \langle \lambda_1 \lambda_2 \rangle
[\tilde{\lambda}_1 \tilde{\lambda}_2].
$$
:::

:::{prf:proof}
By {prf:ref}`prop-twistor-det-lorentz`,

$$
\det P = \frac{1}{2}\epsilon^{AB}\epsilon^{A'B'}P_{AA'}P_{BB'}.
$$

Substitute the decomposition of {prf:ref}`def-twistor-two-system`:

$$
P_{AA'}P_{BB'}
=
\sum_{I,J=1}^2
\lambda_{IA}\tilde{\lambda}_{IA'}
\lambda_{JB}\tilde{\lambda}_{JB'}.
$$

The terms with $I=J$ vanish after contraction with $\epsilon^{AB}\epsilon^{A'B'}$ because

$$
\epsilon^{AB}\lambda_{IA}\lambda_{IB} = 0,
\qquad
\epsilon^{A'B'}\tilde{\lambda}_{IA'}\tilde{\lambda}_{IB'} = 0
$$

by antisymmetry of $\epsilon$. Hence only the cross terms survive:

$$
\det P
= \frac{1}{2}\epsilon^{AB}\epsilon^{A'B'}
\bigl(
\lambda_{1A}\tilde{\lambda}_{1A'}\lambda_{2B}\tilde{\lambda}_{2B'}
+ \lambda_{2A}\tilde{\lambda}_{2A'}\lambda_{1B}\tilde{\lambda}_{1B'}
\bigr).
$$

For the second term, swap $A \leftrightarrow B$ and $A' \leftrightarrow B'$. Each swap contributes
a minus sign from the corresponding $\epsilon$, so the two minus signs cancel. Therefore the second
term equals the first, and the prefactor $\tfrac12$ removes the duplication:

$$
\det P
=
\epsilon^{AB}\lambda_{1A}\lambda_{2B}
\epsilon^{A'B'}\tilde{\lambda}_{1A'}\tilde{\lambda}_{2B'}.
$$

Recognizing the bracket notation from {prf:ref}`def-twistor-spinor-conventions`,

$$
\det P
= \langle \lambda_1 \lambda_2 \rangle
[\tilde{\lambda}_1 \tilde{\lambda}_2].
$$

Combining this with {prf:ref}`prop-twistor-det-lorentz` gives the mass formula. $\square$
:::

:::{prf:corollary} Reality and Positivity on the Lorentzian Slice
:label: cor-twistor-mass-positivity

If $P$ is Hermitian and future-directed timelike, then

$$
[\tilde{\lambda}_1 \tilde{\lambda}_2]
= \overline{\langle \lambda_1 \lambda_2 \rangle}
$$

and therefore

$$
m^2 = \left| \langle \lambda_1 \lambda_2 \rangle \right|^2 \geq 0.
$$
:::

:::{prf:proof}
On the Lorentzian real slice the dotted spinors are complex conjugates of the undotted ones. Hence
the dotted bracket is the complex conjugate of the undotted bracket. The mass formula of
{prf:ref}`thm-twistor-mass-formula` then becomes

$$
m^2
= \langle \lambda_1 \lambda_2 \rangle
\overline{\langle \lambda_1 \lambda_2 \rangle}
= \left| \langle \lambda_1 \lambda_2 \rangle \right|^2,
$$

which is nonnegative. $\square$
:::

:::{prf:proposition} Factorization-Gauge Invariance
:label: prop-twistor-little-group

Assume $P$ is future-directed and Hermitian, so that on the Lorentzian real slice one may write

$$
P = \Lambda \Lambda^\dagger,
\qquad
\Lambda = [\lambda_1\ \lambda_2].
$$

Let $U \in U(2)$ and define a new factorization by

$$
\Lambda' := \Lambda U.
$$

Then

$$
P = \Lambda' \Lambda'^\dagger,
$$

and the bracket transforms by

$$
\langle \lambda'_1 \lambda'_2 \rangle
= \det(U)\,\langle \lambda_1 \lambda_2 \rangle.
$$

Consequently

$$
\left|\langle \lambda'_1 \lambda'_2 \rangle\right|^2
=
\left|\langle \lambda_1 \lambda_2 \rangle\right|^2.
$$

In particular the scalar mass invariant is unchanged by the choice of factorization.
:::

:::{prf:proof}
By construction,

$$
P' = \Lambda' \Lambda'^\dagger
= \Lambda U U^\dagger \Lambda^\dagger
= \Lambda \Lambda^\dagger
= P.
$$

For the bracket, antisymmetry gives

$$
\langle \lambda'_1 \lambda'_2 \rangle
= \det(U)\,\langle \lambda_1 \lambda_2 \rangle,
$$

and similarly

$$
\left|\langle \lambda'_1 \lambda'_2 \rangle\right|^2
= |\det(U)|^2 \left|\langle \lambda_1 \lambda_2 \rangle\right|^2
= \left|\langle \lambda_1 \lambda_2 \rangle\right|^2
$$

because $U$ is unitary. The physical massive little group is the $SU(2)$ subgroup; the extra
overall phase in $U(2)$ is only a factorization gauge. $\square$
:::


(sec-twistor-infinity-mass)=
## 4. Mass Extraction Through the Infinity Twistor

:::{prf:proposition} Infinity-Twistor Mass Identity
:label: prop-twistor-infinity-mass

For a two-twistor system,

$$
m^2
=
\bigl(I_{\alpha\beta} Z_1^\alpha Z_2^\beta\bigr)
\bigl(\bar{I}_{\bar{\alpha}\bar{\beta}}
\bar{Z}_1^{\bar{\alpha}}\bar{Z}_2^{\bar{\beta}}\bigr).
$$
:::

:::{prf:proof}
By {prf:ref}`def-twistor-infinity`,

$$
I_{\alpha\beta} Z_1^\alpha Z_2^\beta
= \langle \lambda_1 \lambda_2 \rangle,
\qquad
\bar{I}_{\bar{\alpha}\bar{\beta}}
\bar{Z}_1^{\bar{\alpha}}\bar{Z}_2^{\bar{\beta}}
= [\tilde{\lambda}_1 \tilde{\lambda}_2].
$$

Multiply the two identities and apply {prf:ref}`thm-twistor-mass-formula`. $\square$
:::

:::{prf:remark}
Proposition {prf:ref}`prop-twistor-infinity-mass` is the precise sense in which the infinity
twistor makes mass visible. The twistor pair itself determines only conformal data; the infinity
twistor chooses the scale that turns the conformal two-form into the physical scalar invariant
$m^2$.
:::


(sec-fractal-set-twistorization)=
## 5. Effective Twistor Operators on Companion Triplets

Sections 1--4 describe the exact twistor formalism for a reconstructed Lorentzian bispinor. The
implemented Fractal Set channels are weaker and more concrete. They do not begin from an on-shell
momentum matrix. Instead they begin from one source walker and its two companion choices.

:::{prf:definition} Source-Frame Companion Triplet
:label: def-effective-twistor-triplet

Fix a recorded source frame $t$ and a source walker $i$. Let

$$
j_t(i) := \operatorname{companions\_distance}_t(i),
\qquad
k_t(i) := \operatorname{companions\_clone}_t(i).
$$

The associated **source-frame companion triplet** is

$$
T_t(i) := (i, j_t(i), k_t(i)).
$$

The twistor companion channels always use both companion types. In particular they are triplet
channels, not pair channels, and they do not depend on the mesonic `pair_selection` choice.
:::

:::{prf:definition} Effective Edge Four-Vectors and Bispinors
:label: def-effective-twistor-edge-data

Work in the physical $D=4$ specialization and retain three spatial coordinates
$a \in \{1,2,3\}$. For an oriented companion edge $(i,j)$ at source frame $t$ define

$$
\Delta x_{ij}^a(t) := x_j^a(t) - x_i^a(t),
\qquad
\Delta v_{ij}^a(t) := v_j^a(t) - v_i^a(t).
$$

Let $\Delta t>0$ denote the algorithmic time step and let $\alpha > 0$ be a fixed velocity scale.
Define the real four-vectors

$$
X_{ij}^\mu(t) := \bigl(\Delta t, \Delta x_{ij}^1(t), \Delta x_{ij}^2(t), \Delta x_{ij}^3(t)\bigr),
$$

$$
V_{ij}^\mu(t) := \bigl(0, \Delta v_{ij}^1(t), \Delta v_{ij}^2(t), \Delta v_{ij}^3(t)\bigr),
$$

and the complex effective bispinor

$$
B_{ij,AA'}(t)
:=
X_{ij}^\mu(t)\sigma_{\mu\,AA'}
+ i \alpha\, V_{ij}^\mu(t)\sigma_{\mu\,AA'}.
$$
:::

The real part of $B_{ij}$ records the edge displacement, while the imaginary part records the
relative velocity weighted by $\alpha$. In the current implementation the dashboard uses the first
three spatial coordinates and $\alpha = 1$.

:::{prf:definition} Effective Edge Twistor
:label: def-effective-edge-twistor

Let $c_{ij}^{(0)}(t)$ and $c_{ij}^{(1)}(t)$ denote the two columns of the matrix $B_{ij}(t)$. Define

$$
n_{ij}^{(r)}(t) := \|c_{ij}^{(r)}(t)\|_2^2,
\qquad r \in \{0,1\}.
$$

Choose the dominant column

$$
c_{ij}^{(*)}(t)
:=
\begin{cases}
c_{ij}^{(0)}(t), & n_{ij}^{(0)}(t) \ge n_{ij}^{(1)}(t), \\
c_{ij}^{(1)}(t), & n_{ij}^{(1)}(t) > n_{ij}^{(0)}(t).
\end{cases}
$$

If $\|c_{ij}^{(*)}(t)\|_2 > \varepsilon$, define the normalized spinor

$$
\lambda_{ij,A}(t)
:=
\frac{c_{ij,A}^{(*)}(t)}{\|c_{ij}^{(*)}(t)\|_2},
$$

and the companion incidence spinor

$$
\mu_{ij}^{A'}(t) := X_{ij}^{AA'}(t)\lambda_{ij,A}(t).
$$

The resulting **effective edge twistor** is

$$
Z_{ij}^\alpha(t) := \bigl(\mu_{ij}^{A'}(t), \lambda_{ij,A}(t)\bigr).
$$

If the chosen column norm is at most $\varepsilon$, or if any index is out of range, or if any of
the walkers in the edge is dead, the edge is declared invalid.
:::

This is a deterministic gauge choice, not the canonical Penrose twistor of a continuum null ray.
Its purpose is only to produce a stable local operator from companion geometry.

:::{prf:definition} Local Twistor Companion Operators
:label: def-effective-twistor-operators

For a valid source-frame triplet $T_t(i) = (i,j_t(i),k_t(i))$, define

$$
\tau_i(t)
:=
I_{\alpha\beta}
Z_{ij_t(i)}^\alpha(t)\,
Z_{ik_t(i)}^\beta(t)
=
\langle \lambda_{ij_t(i)}(t), \lambda_{ik_t(i)}(t) \rangle.
$$

From $\tau_i(t)$ define three scalar local operators:

$$
\mathcal{O}_{\mathrm{S},i}(t) := \Re \tau_i(t),
\qquad
\mathcal{O}_{\mathrm{P},i}(t) := \Im \tau_i(t),
\qquad
\mathcal{O}_{\mathrm{G},i}(t) := |\tau_i(t)|^2.
$$

Next define the complex Pauli-vector bilinear

$$
W_i^a(t)
:=
\lambda_{ij_t(i)}(t)^\dagger \sigma^a \lambda_{ik_t(i)}(t),
\qquad a \in \{1,2,3\}.
$$

Its real and imaginary parts define the implemented spin-one operators

$$
\mathcal{O}_{\mathrm{V},i}^a(t) := \Re W_i^a(t),
\qquad
\mathcal{O}_{\mathrm{A},i}^a(t) := \Im W_i^a(t).
$$

Finally define the five real spin-two-like components

$$
Q_{xy,i}(t) := \Re\!\bigl(W_i^x(t)W_i^y(t)\bigr),
\qquad
Q_{xz,i}(t) := \Re\!\bigl(W_i^x(t)W_i^z(t)\bigr),
\qquad
Q_{yz,i}(t) := \Re\!\bigl(W_i^y(t)W_i^z(t)\bigr),
$$

$$
Q_{x^2-y^2,i}(t) := \frac{1}{\sqrt{2}}\Re\!\bigl(W_i^x(t)^2 - W_i^y(t)^2\bigr),
$$

$$
Q_{2z^2-x^2-y^2,i}(t)
:=
\frac{1}{\sqrt{6}}
\Re\!\bigl(2W_i^z(t)^2 - W_i^x(t)^2 - W_i^y(t)^2\bigr),
$$

and contract them to the scalar dashboard tensor observable

$$
\mathcal{O}_{\mathrm{T},i}(t)
:=
\frac{1}{5}
\Bigl(
Q_{xy,i}(t)+Q_{xz,i}(t)+Q_{yz,i}(t)+Q_{x^2-y^2,i}(t)+Q_{2z^2-x^2-y^2,i}(t)
\Bigr).
$$

The labels S, P, G, V, A, and T stand respectively for scalar, pseudoscalar,
glueball-like scalar, vector, axial-vector, and tensor-like companion operators.
No baryon-valued twistor companion operator is implemented.
:::

:::{prf:proposition} Positivity of the Glueball-Like Operator
:label: prop-effective-twistor-glueball-positive

For every valid source-frame triplet,

$$
\mathcal{O}_{\mathrm{G},i}(t)
=
\mathcal{O}_{\mathrm{S},i}(t)^2
+
\mathcal{O}_{\mathrm{P},i}(t)^2
\ge 0.
$$
:::

:::{prf:proof}
Write $\tau_i(t) = a_i(t) + i b_i(t)$ with $a_i,b_i \in \mathbb{R}$. By definition,

$$
\mathcal{O}_{\mathrm{S},i}(t) = a_i(t),
\qquad
\mathcal{O}_{\mathrm{P},i}(t) = b_i(t),
\qquad
\mathcal{O}_{\mathrm{G},i}(t) = |\tau_i(t)|^2.
$$

But

$$
|\tau_i(t)|^2 = (a_i(t) + i b_i(t))(a_i(t) - i b_i(t)) = a_i(t)^2 + b_i(t)^2.
$$

Therefore

$$
\mathcal{O}_{\mathrm{G},i}(t)
=
\mathcal{O}_{\mathrm{S},i}(t)^2
+
\mathcal{O}_{\mathrm{P},i}(t)^2
\ge 0.
$$

$\square$
:::


(sec-twistor-vs-euclidean)=
## 6. Companion-Tracking Correlators and Spectral Masses

The local quantities of {prf:ref}`def-effective-twistor-operators` are not yet masses. They become
spectral observables only after being inserted into the same Euclidean correlator machinery already
used in {doc}`09_qft_calibration`.

:::{prf:definition} Source-Frame Twistor Correlators
:label: def-effective-twistor-correlators

Fix one of the local operators
$\mathcal{O}_{X,i}(t)$ with
$X \in \{\mathrm{S}, \mathrm{P}, \mathrm{G}, \mathrm{T}\}$,
or one of the vector-valued operators
$\mathcal{O}_{Y,i}^a(t)$ with
$Y \in \{\mathrm{V}, \mathrm{A}\}$. For each source time
$t$ and source walker $i$, keep the source-frame triplet $T_t(i) = (i, j_t(i), k_t(i))$ fixed.
For a lag $\ell \ge 0$, evaluate the sink operator at time $t+\ell$ using the same source indices:

$$
\mathcal{O}^{(\ell)}_{X,i}(t)
:=
\mathcal{O}_X\bigl(t+\ell; i, j_t(i), k_t(i)\bigr).
$$

Let $\mathbf{1}_{X,t,\ell}(i)$ be the indicator that both the source triplet and the corresponding
sink evaluation are valid. Define

$$
N_X(\ell)
:=
\sum_{t=0}^{T-1-\ell}\sum_i \mathbf{1}_{X,t,\ell}(i).
$$

Whenever $N_X(\ell) > 0$, the raw correlator is

$$
C_X^{\mathrm{raw}}(\ell)
:=
\frac{1}{N_X(\ell)}
\sum_{t=0}^{T-1-\ell}\sum_i
\mathbf{1}_{X,t,\ell}(i)\,
\mathcal{O}_{X,i}(t)\,
\mathcal{O}^{(\ell)}_{X,i}(t).
$$

Let $\overline{\mathcal{O}}_X$ denote the mean of $\mathcal{O}_{X,i}(t)$ over valid source
triplets. The connected correlator is

$$
C_X^{\mathrm{conn}}(\ell)
:=
\frac{1}{N_X(\ell)}
\sum_{t=0}^{T-1-\ell}\sum_i
\mathbf{1}_{X,t,\ell}(i)\,
\bigl(\mathcal{O}_{X,i}(t)-\overline{\mathcal{O}}_X\bigr)
\bigl(\mathcal{O}^{(\ell)}_{X,i}(t)-\overline{\mathcal{O}}_X\bigr).
$$

For the vector and axial-vector channels, replace the pointwise product by the Euclidean
dot product in $\mathbb{R}^3$:

$$
C_Y^{\mathrm{raw}}(\ell)
:=
\frac{1}{N_Y(\ell)}
\sum_{t=0}^{T-1-\ell}\sum_i
\mathbf{1}_{Y,t,\ell}(i)\,
\mathcal{O}_{Y,i}(t)\cdot\mathcal{O}^{(\ell)}_{Y,i}(t),
$$

$$
C_Y^{\mathrm{conn}}(\ell)
:=
\frac{1}{N_Y(\ell)}
\sum_{t=0}^{T-1-\ell}\sum_i
\mathbf{1}_{Y,t,\ell}(i)\,
\bigl(\mathcal{O}_{Y,i}(t)-\overline{\mathcal{O}}_Y\bigr)
\cdot
\bigl(\mathcal{O}^{(\ell)}_{Y,i}(t)-\overline{\mathcal{O}}_Y\bigr).
$$
:::

Definition {prf:ref}`def-effective-twistor-correlators` is exactly the source-frame companion
tracking rule used by the implemented channel code. The sink triplet is **not** rebuilt from sink
companions. The source companions are propagated through the lag.

:::{prf:proposition} The Local Twistor Operators Are Not Particle Masses
:label: prop-effective-twistor-not-masses

The pointwise quantities
$\mathcal{O}_{\mathrm{S},i}(t)$,
$\mathcal{O}_{\mathrm{P},i}(t)$,
$\mathcal{O}_{\mathrm{G},i}(t)$,
$\mathcal{O}_{\mathrm{V},i}^a(t)$,
$\mathcal{O}_{\mathrm{A},i}^a(t)$, and
$\mathcal{O}_{\mathrm{T},i}(t)$
are local operator values on the companion graph. They are not particle masses.
:::

:::{prf:proof}
By definition, each local twistor companion operator depends only on one source walker, its two
companion choices, and the geometric data at one recorded time. It is therefore a local field
observable on the Fractal Set.

By contrast, {doc}`09_qft_calibration` defines a channel mass through the asymptotic decay rate of a
two-point function, equivalently through the spectral poles or the correlation length of that
channel. A mass is therefore an attribute of the long-time correlator, not of a single local value
at one time and one source triplet.

Hence the local twistor companion operators are not masses. They are operator insertions whose
correlators may couple to massive states. $\square$
:::

:::{prf:theorem} Spectral Meaning of the Twistor Companion Channels
:label: thm-effective-twistor-spectral-meaning

Assume the Euclidean transfer-matrix/spectral framework of {doc}`09_qft_calibration`. Let
$\widehat{\mathcal{O}}_X(t)$ denote the frame-averaged twistor operator associated with one of the
scalar-valued families
$X \in \{\mathrm{S}, \mathrm{P}, \mathrm{G}, \mathrm{T}\}$,
or let $\widehat{\mathcal{O}}_Y(t)$ denote the frame-averaged vector-valued operator associated with
$Y \in \{\mathrm{V}, \mathrm{A}\}$. Then the connected two-point function has
the spectral form

$$
C_X^{\mathrm{conn}}(\ell)
=
\sum_{n>0}
\left|\langle n | \widehat{\mathcal{O}}_X | 0 \rangle\right|^2
e^{-E_n \ell \Delta t}.
$$

for scalar-valued $X$, and analogously with the Euclidean dot product for $Y \in \{\mathrm{V},\mathrm{A}\}$.

If at least one overlap is nonzero, then for large $\ell$

$$
C_X^{\mathrm{conn}}(\ell)
\sim
\left|\langle n_X | \widehat{\mathcal{O}}_X | 0 \rangle\right|^2
e^{-E_{n_X} \ell \Delta t},
$$

where $E_{n_X}$ is the smallest energy with nonzero overlap. Therefore the plateau mass extracted
from the twistor companion correlator is the mass of the lightest state that couples to that
operator.
:::

:::{prf:proof}
Let $T = e^{-\Delta t\,H}$ be the Euclidean transfer operator and let
$\{|n\rangle\}_{n \ge 0}$ be a complete orthonormal basis of energy eigenstates with
$H|n\rangle = E_n |n\rangle$ and $E_0 = 0$ for the vacuum.

For the frame-averaged operator $\widehat{\mathcal{O}}_X$, the unconnected correlator is

$$
\langle 0 | \widehat{\mathcal{O}}_X(0)\,\widehat{\mathcal{O}}_X(\ell) | 0 \rangle
=
\langle 0 | \widehat{\mathcal{O}}_X\, T^\ell \,\widehat{\mathcal{O}}_X | 0 \rangle.
$$

Insert the identity $\sum_n |n\rangle\langle n| = \mathbf{1}$ between the two operators:

$$
\langle 0 | \widehat{\mathcal{O}}_X\, T^\ell \,\widehat{\mathcal{O}}_X | 0 \rangle
=
\sum_n
\langle 0 | \widehat{\mathcal{O}}_X | n \rangle
\langle n | \widehat{\mathcal{O}}_X | 0 \rangle
e^{-E_n \ell \Delta t}.
$$

Since
$\langle 0 | \widehat{\mathcal{O}}_X | n \rangle
= \overline{\langle n | \widehat{\mathcal{O}}_X | 0 \rangle}$,
this becomes

$$
\sum_n
\left|\langle n | \widehat{\mathcal{O}}_X | 0 \rangle\right|^2
e^{-E_n \ell \Delta t}.
$$

Subtracting the vacuum piece gives the connected correlator, so the $n=0$ term is removed and

$$
C_X^{\mathrm{conn}}(\ell)
=
\sum_{n>0}
\left|\langle n | \widehat{\mathcal{O}}_X | 0 \rangle\right|^2
e^{-E_n \ell \Delta t}.
$$

Let $n_X$ be the smallest index with nonzero overlap. Then every other surviving term has strictly
larger exponential suppression for large $\ell$, so the asymptotics are dominated by the
$n_X$-term, proving the claim. $\square$
:::

The channel interpretations are then the expected ones:

- $\widehat{\mathcal{O}}_{\mathrm{S}}$ is a scalar operator;
- $\widehat{\mathcal{O}}_{\mathrm{P}}$ is a pseudoscalar operator;
- $\widehat{\mathcal{O}}_{\mathrm{G}}$ is a positive scalar operator and is the natural
  glueball-like twistor probe.


## 7. Relation to the Exact Twistor Mass Formula

The continuum twistor identity of Sections 1--4 is still valid, but it applies only after a
separate Lorentzian momentum bispinor has been reconstructed for a channel or event. That is a
second step beyond the implemented twistor companion operators.

:::{prf:proposition} Compatibility with On-Shell Channel Masses
:label: prop-twistor-correlator-compatibility

Let $m_\chi$ be the mass of a Fractal Set channel extracted from Euclidean correlators, possibly
using one of the twistor companion operators above. Suppose the same channel admits an
Osterwalder-Schrader reconstructed Lorentzian momentum $p_\chi^\mu$ satisfying

$$
p_{\chi,\mu} p_\chi^\mu = m_\chi^2.
$$

Let $P_{\chi,AA'} := p_\chi^\mu \sigma_{\mu\,AA'}$ be the associated Hermitian bispinor. Then

$$
\det P_\chi = m_\chi^2,
$$

and if

$$
P_{\chi,AA'}
= \lambda_{1A}\tilde{\lambda}_{1A'} + \lambda_{2A}\tilde{\lambda}_{2A'}
$$

is any two-spinor factorization, then

$$
m_\chi^2
= \langle \lambda_1 \lambda_2 \rangle
[\tilde{\lambda}_1 \tilde{\lambda}_2].
$$
:::

:::{prf:proof}
By {prf:ref}`def-twistor-momentum-map`,

$$
P_{\chi,AA'} := p_\chi^\mu \sigma_{\mu\,AA'}
$$

is the bispinor corresponding to the reconstructed Lorentzian momentum. By
{prf:ref}`prop-twistor-det-lorentz`,

$$
\det P_\chi = p_{\chi,\mu} p_\chi^\mu = m_\chi^2.
$$

If a two-spinor factorization of $P_\chi$ is chosen, then {prf:ref}`thm-twistor-mass-formula`
gives

$$
\det P_\chi
= \langle \lambda_1 \lambda_2 \rangle
[\tilde{\lambda}_1 \tilde{\lambda}_2].
$$

Combining the two equalities yields the claim. $\square$
:::

:::{prf:remark}
This proposition is the correct division of labor.

- The implemented twistor companion channels define new operator families and extract masses from
  their Euclidean correlators.
- The exact twistor mass formula computes a Lorentzian invariant only after a channel has been
  reconstructed on shell as a momentum bispinor.

They are compatible, but they are not the same procedure. The current code implements the first,
not the second.
:::
