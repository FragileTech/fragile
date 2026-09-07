# Direct Field Observables and Lattice QFT on the Fractal Set

(sec-lqft-intro)=
## 1. Recorded Graphs and Field Observables

:::{div} feynman-prose
Start with a recorded Fractal Set. Its reconstruction gives the numerical
inputs from which we calculate color overlaps, determinants, and companion
doublet contractions. We can describe a configuration through these invariant
coordinates: the full collection identifies its internal symmetry orbit
exactly. Passing the recorded law through the same map preserves every
integrable correlation of these observables. Retaining the complete record
also lets us compute conditional expectations with the actual update kernel.
The field representation then carries the recorded evolution as well as the
measurements.

The reconstruction theorems identify the inputs; the LSI estimates control
the resulting observables through their reconstruction maps. Each result
keeps its specified measure and domain. On centered record modes, exterior
products faithfully represent alternating insertions into replica
observables. The recorded transition propagates these modes and determines
a completely positive evolution of their CAR operators, with explicit
covariance calculations. Every expectation uses the recorded law, and
every transition uses the complete algorithmic update.
:::

:::{prf:definition} Data and conventions for the field constructions
:label: def-lqft-scope

Let $\mathcal F$ be a finite recorded Fractal Set with episode vertices and
CST, IG, and IA edges as in {doc}`01_fractal_set`. Choose oriented loops or
faces from its interaction complex when defining gauge actions. Traversing
an edge backwards uses its inverse transport; it does not add a backwards
CST causal relation. The finite CST order is the one proved in
{prf:ref}`thm-fractal-is-causal-set`.

Write $N$ for the number of observations in a sampling statement and $n$ for
an internal gauge-fiber dimension. These numbers are independent. A spatial
geometry $(X,g_R)$ and a Lorentzian comparison geometry, when used, are
specified separately with the conditions of
{prf:ref}`assm-cst-continuum-geometry`. Describing a recorded edge as timelike
or spacelike requires that comparison; its graph type alone is combinatorial.

The field observables below are functions of the recorded Fractal Set.
Their expectations use its recorded law, and their evolution uses its
complete update kernel.
:::

:::{prf:definition} Direct fields and their invariant representation
:label: def-lqft-direct-field-route

The direct formulation starts from the complete recorded Fractal Set and
the numerical fields reconstructed from it. Its color contractions,
companion amplitudes, doublets, determinants, and triangle products are
defined in {prf:ref}`def-sm-direct-color-contractions` and
{prf:ref}`def-sm-direct-companion-doublet`. Their law is
{prf:ref}`def-sm-direct-observable-law`, with the original record measure.
Theorems {prf:ref}`thm-sm-direct-orbit-isomorphism` and
{prf:ref}`thm-sm-direct-measure-isomorphism` identify the full invariant
coordinates with the $SU(3)$ and $SU(2)$ orbit spaces and their observable
Hilbert spaces. This identification uses numerical contractions and
requires no Dirac matrices.

The Fractal Set reconstruction and the LSI estimates enter this formulation
through {prf:ref}`thm-sm-direct-existing-machinery`. In particular, the
finite representation change preserves the direct field correlations
exactly. A normalized overlap triangle is the composite observable of
{prf:ref}`prop-sm-direct-triangle-projectors`. Its ordered rank-one
projectors and scalar phase are evaluated from the recorded color data.
:::

(sec-gauge-theory)=
### Recorded link orientation and loop readouts

:::{prf:definition} Orientation of reconstructed comparison links
:label: def-lqft-link-convention

For the recorded gauge transports of
{prf:ref}`def-fractal-set-gauge-connection`, write the comparison link as
$U_{ij}:V_j\to V_i$. Reverse traversal uses
$U_{ji}=U_{ij}^{-1}=U_{ij}^{\dagger}$. Changes of the recorded fiber coordinates obey

$$
\psi_i\mapsto\Omega_i\psi_i,\qquad
U_{ij}\mapsto\Omega_iU_{ij}\Omega_j^{-1}.
$$

Thus $U_{ij}\psi_j-\psi_i$ is expressed in the frame at $i$.
Every link in a product is evaluated by the existing recorded reconstruction.
:::

:::{prf:definition} Recorded companion-doublet transport
:label: def-su2-clone-transport

Use the companion doublet of {prf:ref}`def-sm-direct-companion-doublet`
and its recorded attribution transport from
{prf:ref}`def-fractal-set-gauge-connection`. Its comparison matrices use
{prf:ref}`def-lqft-link-convention` on the corresponding pair-state fibers.
Both the doublet and its transport are evaluated from the recorded pair data.
:::

:::{prf:definition} Recorded Wilson loop observable
:label: def-wilson-loop-lqft

For a closed recorded path $\gamma=(i_0,\ldots,i_m=i_0)$ with matching
comparison fibers, set

$$
U_\gamma=U_{i_0i_1}\cdots U_{i_{m-1}i_0},\qquad
W[\gamma]=\operatorname{Tr}U_\gamma,\qquad
w[\gamma]=\frac1n\operatorname{Tr}U_\gamma.
$$

Adjacent frame changes cancel, leaving conjugation at $i_0$, so the traces
are gauge invariant. Unitarity gives $|w|\le1$. Reverse traversal
conjugates the trace. Expectations are integrals of these recorded
readouts under the original record law.
:::

(sec-fg-lqft-wilson-loops)=
## 2. Recorded Diagram Subcomplexes

:::{prf:definition} Recorded diagram subcomplex
:label: def-fractal-set-feynman-diagram

A diagram $\mathcal D=(E_{\mathcal D},\mathcal T_{\mathcal D})$ is a finite
connected oriented subcomplex with edges in
$E_{\mathrm{CST}}\cup E_{\mathrm{IG}}\cup E_{\mathrm{IA}}$ and selected
interaction triangles. Every selected triangle has its recorded CST, IG,
and IA edges; CST orientations are future directed. CST edges incident to
one selected triangle may be designated external legs, and those incident
to two selected triangles internal legs. Other incidences require their own
convention. Clone ancestry remains distinct from the CST order.

This is a combinatorial diagram definition. Identifying these diagrams
with terms of a perturbative field expansion requires a chosen action,
propagators, vertex tensors, and combinatorial factors.
:::

:::{prf:definition} Specified diagram weights
:label: def-fractal-set-diagram-weight

For recorded scalar edge readouts $G_a$ and scalar triangle readouts $W_\triangle$,
set

$$
\mathcal A(\mathcal D)=\prod_{a\in E_{\mathcal D}}G_a
\prod_{\triangle\in\mathcal T_{\mathcal D}}W_\triangle.
$$

For matrix or spinor weights, specify index contractions and orderings
instead of an unordered product. Gauge-invariant amplitudes contract every
internal fiber index consistently and state how external indices transform.
A sum over recorded subcomplexes is a finite statistic once these rules are
given; it equals a field path-integral expansion only after that identity
has been established for the selected action.
:::

(sec-fermions)=
## 3. Cloning Antisymmetry and Recorded Fermionic Operators

:::{div} feynman-prose
The direct companion observables use numerical cloning scores. Exchanging
the two walkers reverses the fitness difference, while the denominator
changes with the receiving walker. Keeping both denominators gives the exact
weighted antisymmetry proved below. These directed scores supply the
coefficients for the finite exterior-algebra action constructed first.

To connect fermionic operators to measured dynamics, we then start from the
stationary record process itself. Think of a mode as a centered,
square-integrable measurement of a complete swarm state. Its transition
operator tells us how the conditional mean of that measurement changes with
time. The construction in {ref}`sec-lqft-recorded-fermionic-reconstruction`
lifts this actual operator to exterior products of modes. Creation and
contraction satisfy the canonical anticommutation relations there, and the
one-mode matrix element is exactly the recorded covariance.

What does an exterior product measure? Run independent copies of the whole
swarm and take a normalized determinant of their mode values. The proof
identifies these centered antisymmetric replica sectors unitarily with the
exterior sectors and intertwines their transitions. Higher-sector matrix
elements are therefore exact determinants of recorded covariances. Each
replica retains all its internal walker interactions; the independence is
between complete simulations. The theorem concerns these replica observables
and their sector-changing operators, and does not identify the walkers within
one simulation with fermionic particles.

The word calculation below extends this correspondence to finite sequences
of insertions, contractions, and recorded transitions. Alternating insertion
has a faithful exterior algebra; stochastic replacement retains its positive
probability-kernel composition. The same recorded transition also gives a
completely positive evolution of the CAR algebra, including when the
algorithmic dynamics is irreversible.
:::

### Exact consequences of the score

:::{prf:theorem} Weighted antisymmetry of cloning scores
:label: thm-cloning-antisymmetry-lqft

For fixed fitnesses $V_i$ with $V_i+\varepsilon_{\mathrm{clone}}>0$, let

$$
S_i(j)=\frac{V_j-V_i}{V_i+\varepsilon_{\mathrm{clone}}}.
$$

Then

$$
(V_i+\varepsilon_{\mathrm{clone}})S_i(j)
=-(V_j+\varepsilon_{\mathrm{clone}})S_j(i)=V_j-V_i.
$$

In addition,

$$
S_i(j)+S_j(i)=
\frac{(V_j-V_i)^2}
{(V_i+\varepsilon_{\mathrm{clone}})(V_j+\varepsilon_{\mathrm{clone}})}.
$$

Thus the weighted scores are exactly antisymmetric, while the raw scores
are generally not. If $V_i=V_j$, both vanish. If both denominators are at
least $v_*>0$, their failure of antisymmetry is at most
$|V_i-V_j|^2/v_*^2$.
:::

:::{prf:proof}
Multiply the definitions by their denominators for the first identity.
For the second, put the two fractions over the common denominator and
simplify the numerator to $(V_j-V_i)^2$. Positivity of the denominators
proves the final bound. $\square$
:::

:::{prf:definition} Antisymmetrized score kernel
:label: def-fermionic-kernel-lqft

For a specified real score matrix $K$, define

$$
\widetilde K=K-K^\top,\qquad
\widetilde K_{ij}=-\widetilde K_{ji},\qquad\widetilde K_{ii}=0.
$$

For $K_{ij}=S_i(j)$, this gives

$$
\widetilde K_{ij}=(V_j-V_i)
\left(\frac1{V_i+\varepsilon_{\mathrm{clone}}}
+\frac1{V_j+\varepsilon_{\mathrm{clone}}}\right).
$$

It is an antisymmetric matrix available as a coupling in an auxiliary
fermionic action. Antisymmetry alone does not determine a propagator,
a dispersion relation, a Clifford representation, or exchange statistics
of the sampled particles.
:::

:::{prf:theorem} Pairwise exclusion of opposing cloning directions
:label: thm-exclusion-principle

For the stated positive-denominator scores, if a cloning direction is
eligible only when its score is strictly positive, at most one of
$i\leftarrow j$ and $j\leftarrow i$ is eligible for a fixed pair of
fitness values. When $V_i=V_j$, neither is eligible.

This conclusion concerns opposing score-based directions at the same
state. Companion sampling, Bernoulli acceptance, and forced revival are
separate parts of the update. It is not a bound on the number of walkers
occupying the same spatial or quantum state.
:::

:::{prf:proof}
The sign of $S_i(j)$ is the sign of $V_j-V_i$, and the reverse score has the
opposite sign. If the fitnesses agree, both scores are zero. $\square$
:::

### Exterior algebra of recorded modes

:::{prf:definition} Exterior symbols of recorded modes
:label: post-grassmann

For a finite mode space $E$ of recorded observables, quotient zero-norm
linear combinations and choose a basis $e_1,\ldots,e_m$. Its exterior
algebra is the canonical quotient

$$
\Lambda(E)=T(E)/\langle f\otimes f:f\in E\rangle.
$$

The image $\psi_i$ of $e_i$ satisfies
$\psi_i\psi_j=-\psi_j\psi_i$ and $\psi_i^2=0$.
The faithful identification with the existing alternating record
insertions is proved in {prf:ref}`thm-lqft-oriented-word-algebra`.
Thus the record insertion representation derives these relations.

:::

(sec-lqft-recorded-fermionic-reconstruction)=
### Fermionic reconstruction of recorded transition correlations

:::{prf:definition} Centered record space and its exterior sectors
:label: def-lqft-record-fock-space

Use an actual conservative stationary record process with invariant law $\pi$
and transition operators $P_t$, with the law distinction of
{prf:ref}`def-sm-direct-observable-law`. In discrete time write $P^m$ instead.
Set $\mathcal H=L^2_0(\pi;\mathbb C)$, with inner product
$\langle f,g\rangle=\int\overline f g\,d\pi$. The states underlying $\pi$
are complete Markov states; a mode $f\in\mathcal H$ can depend on the entire
swarm. It is not identified with an individual walker.

On exterior products use

$$
\langle f_1\wedge\cdots\wedge f_k,
        g_1\wedge\cdots\wedge g_k\rangle
=\det[\langle f_i,g_j\rangle]_{i,j=1}^k.
$$

Quotient zero-norm vectors and complete to obtain $\Lambda^k\mathcal H$.
The fermionic Hilbert space is
$\mathcal F_-(\mathcal H)=\bigoplus_{k\ge0}\Lambda^k\mathcal H$,
with vacuum $\Omega=1\in\Lambda^0\mathcal H=\mathbb C$.
This representation uses complex Hilbert spaces and exterior products;
Dirac matrices play no role.
:::

:::{prf:theorem} CAR operators and the lift of the recorded dynamics
:label: thm-lqft-record-fock-reconstruction

On the space of {prf:ref}`def-lqft-record-fock-space`, define

$$
\begin{aligned}
a^\dagger(f)(g_1\wedge\cdots\wedge g_k)
 &=f\wedge g_1\wedge\cdots\wedge g_k,\\
a(f)(g_1\wedge\cdots\wedge g_k)
 &=\sum_{r=1}^k(-1)^{r-1}\langle f,g_r\rangle
       g_1\wedge\cdots\widehat{g_r}\cdots\wedge g_k,
\qquad a(f)\Omega=0.
\end{aligned}
$$

These extend to bounded adjoint operators with
$\|a(f)\|=\|a^\dagger(f)\|=\|f\|$, and satisfy

$$
\{a(f),a^\dagger(g)\}=\langle f,g\rangle I,\qquad
\{a(f),a(g)\}=\{a^\dagger(f),a^\dagger(g)\}=0.
$$

The actual recorded transition has the contraction lift

$$
\Gamma_-(P_t)=\bigoplus_{k\ge0}\Lambda^k(P_t|_{\mathcal H}),
\quad
\Lambda^kP_t(g_1\wedge\cdots\wedge g_k)
=P_tg_1\wedge\cdots\wedge P_tg_k.
$$

It preserves the vacuum and obeys the semigroup law. Strong continuity of
$P_t$ gives strong continuity of its lift. Its matrix elements satisfy

$$
\begin{aligned}
\langle\Omega,a(f)\Gamma_-(P_t)a^\dagger(g)\Omega\rangle
 &=\langle f,P_tg\rangle
 =\mathbb E_\pi[\overline{f(S_0)}g(S_t)],\\
\langle f_1\wedge\cdots\wedge f_k,
 \Gamma_-(P_t)(g_1\wedge\cdots\wedge g_k)\rangle
 &=\det[\mathbb E_\pi\overline{f_i(S_0)}g_j(S_t)]_{i,j=1}^k.
\end{aligned}
$$

The first expression is a connected covariance because the modes are
centered. The vacuum state is $\omega(A)=\langle\Omega,A\Omega\rangle$.
Thus the algebra, positive state, transition operators, and the displayed
correspondence with recorded statistics are all specified.
:::

:::{prf:proof}
**Exterior norm and adjoints.** For finitely many vectors, choose an
orthonormal basis of their span. In that basis the coefficient of the basis
wedge $e_{i_1}\wedge\cdots\wedge e_{i_k}$ is the corresponding coordinate
minor. Expanding the determinant of the Gram matrix gives the sum of the
squared moduli of these minors. This proves positivity of the displayed
inner product; equivalently the determinant expansion in
{prf:ref}`thm-lqft-replica-isomorphism` below supplies that calculation.
Basis wedges are orthonormal. Inserting a basis vector at the front and
sorting it into increasing order has exactly the sign of deleting it in
the contraction formula. Therefore $a(f)$ is the adjoint of $a^\dagger(f)$
on finite exterior sums.

**Anticommutators.** Inserting $g$ as the first vector in the contraction
formula gives, term by term,

$$
a(f)(g\wedge\eta)=\langle f,g\rangle\eta-g\wedge a(f)\eta.
$$

This proves the mixed relation. Two insertions change sign when exchanged,
so their anticommutator is zero. For two deletions at positions $r<s$, the
sign for deleting $r$ and then the original $s$ is
$(-1)^{r-1+s-2}$; the reverse deletion order has sign
$(-1)^{s-1+r-1}$. The signs are opposite. Pairing the two orders for every
pair of positions proves the contraction anticommutator is zero.

For a finite exterior sum $\eta$, the mixed identity at $g=f$ gives

$$
\|a(f)\eta\|^2+\|a^\dagger(f)\eta\|^2
=\|f\|^2\|\eta\|^2.
$$

Both operators therefore extend with norm at most $\|f\|$.
Since $a^\dagger(f)\Omega=f$ and $\|\Omega\|=1$, equality holds for
creation and hence for its adjoint. All CAR identities extend by continuity.
In particular, for a unit mode $f$, $n_f=a^\dagger(f)a(f)$ satisfies

$$
n_f^2=a^\dagger(f)(I-a^\dagger(f)a(f))a(f)=n_f,
$$

because the two repeated creation or annihilation operators square to zero.
Thus this mode has occupations zero and one in the constructed space.

**Transition lift.** Conditional Jensen and stationarity give

$$
\|P_tf\|_2^2
\le\int P_t|f|^2\,d\pi=\|f\|_2^2,
\qquad \pi(P_tf)=\pi f.
$$

Consequently $P_t$ restricts to a contraction on $\mathcal H$.
On the antisymmetric tensor realization, $\Lambda^kP_t$ is the restriction
of $P_t^{\otimes k}$ and has norm at most $\|P_t\|^k\le1$.
The tensor power commutes with every permutation, so it preserves the
antisymmetric subspace. The direct sum, with identity on the vacuum, is a
contraction. Applying two lifts to a decomposable wedge proves
$\Gamma_-(P_t)\Gamma_-(P_s)=\Gamma_-(P_{t+s})$ on a dense set and hence
everywhere. Finite products of strongly convergent bounded operators give
strong continuity on each decomposable wedge. For an arbitrary Fock vector,
truncate the sector sum, approximate each retained sector by such wedges,
and use the uniform contraction bound to control the discarded norm.

**State and correlations.** For every bounded $A$,
$\omega(A^*A)=\|A\Omega\|^2\ge0$ and $\omega(I)=1$.
Creation maps the vacuum to $g$ in the one-particle sector; the lift maps it
to $P_tg$; contraction maps this to $\langle f,P_tg\rangle\Omega$.
The Markov conditional expectation gives

$$
\langle f,P_tg\rangle
=\int\overline{f(s)}\,\mathbb E[g(S_t)\mid S_0=s]d\pi(s)
=\mathbb E_\pi[\overline{f(S_0)}g(S_t)].
$$

For $k$ vectors, the definition of the exterior inner product after the
transition gives the displayed determinant. In creation/annihilation
notation its left side is

$$
\omega\!\left(a(f_k)\cdots a(f_1)\Gamma_-(P_t)
                 a^\dagger(g_1)\cdots a^\dagger(g_k)\right),
$$

with the reversed annihilator order coming from taking the adjoint of the
created bra vector. $\square$
:::

:::{prf:proposition} Two-time reconstruction for a finite recorded law
:label: prop-lqft-finite-record-transfer

For any actual joint law of recorded variables $(Y_s,Y_t)$, let
$\mu_s,\mu_t$ be their marginal laws and
$\mathcal H_s=L^2_0(\mu_s)$, $\mathcal H_t=L^2_0(\mu_t)$.
Conditional expectation defines a contraction

$$
T_{s,t}:\mathcal H_t\longrightarrow\mathcal H_s,
\qquad (T_{s,t}g)(Y_s)=\mathbb E[g(Y_t)\mid Y_s].
$$

Its exterior lift maps $\mathcal F_-(\mathcal H_t)$ to
$\mathcal F_-(\mathcal H_s)$ and satisfies

$$
\langle\Omega_s,a_s(f)\Gamma_-(T_{s,t})a_t^\dagger(g)\Omega_t\rangle
=\mathbb E[\overline{f(Y_s)}g(Y_t)].
$$

The $k$-sector matrix element is the determinant of these two-time
covariances, equivalently the antisymmetrized expectation under $k$
independent copies of this joint law. This applies to finite observation
windows, masks, and the declared time alignment of the direct channels,
without replacing their law by an equilibrium law. A finite empirical joint
law also gives the exact same algebraic identities for its empirical means.

For a Markov family of recorded states the operators compose as
$T_{r,s}T_{s,t}=T_{r,t}$. For a compressed non-Markov readout, the two-time
construction is still defined; that composition is not asserted.
:::

:::{prf:proof}
For centered $g$, total expectation gives
$\mu_s(T_{s,t}g)=\mu_t g=0$. Conditional Jensen gives

$$
\|T_{s,t}g\|_{L^2(\mu_s)}^2
\le\mathbb E\mathbb E[|g(Y_t)|^2\mid Y_s]
=\|g\|_{L^2(\mu_t)}^2.
$$

Therefore the contraction and exterior-lift calculations of
{prf:ref}`thm-lqft-record-fock-reconstruction` apply between the two Hilbert
spaces. The vacuum-to-one-particle calculation is

$$
\langle f,T_{s,t}g\rangle_{\mu_s}
=\mathbb E[\overline{f(Y_s)}\mathbb E[g(Y_t)\mid Y_s]]
=\mathbb E[\overline{f(Y_s)}g(Y_t)].
$$

Taking the exterior inner product gives its determinant. Expanding that
determinant and using independent copies of the same two-time law gives
the replica formula exactly as in
{prf:ref}`thm-lqft-replica-isomorphism`, with product marginals at the two
ends. None of these steps uses stationarity, a continuum limit, or a fitted
spectral model. For empirical laws all integrals are finite weighted sums;
this is an exact identity of estimators, rather than a statistical guarantee
that the empirical law equals the population law.

For the composition assertion, the Markov property gives
$\mathbb E[g(Y_t)\mid Y_r,Y_s]=\mathbb E[g(Y_t)\mid Y_s]$ for
$r\le s\le t$. Taking conditional expectation with respect to $Y_r$
then gives
$\mathbb E[(T_{s,t}g)(Y_s)\mid Y_r]
=\mathbb E[g(Y_t)\mid Y_r]$, which is the stated operator identity.
$\square$
:::

:::{prf:theorem} Exact isomorphism with centered antisymmetric replica sectors
:label: thm-lqft-replica-isomorphism

Let $\mathscr H_k\subset L^2(\pi^{\otimes k})$ be the closed span of
products $f_1(s_1)\cdots f_k(s_k)$ with all $f_i\in L^2_0(\pi)$.
Let $\mathscr H_k^-$ be its antisymmetric subspace. Then

$$
J_k(f_1\wedge\cdots\wedge f_k)(s_1,\ldots,s_k)
=\frac1{\sqrt{k!}}\det[f_i(s_j)]_{i,j=1}^k
$$

extends to a unitary map $\Lambda^k\mathcal H\to\mathscr H_k^-$.
It intertwines the lifted transition with the transition of $k$ independent
copies of the complete record process:

$$
J_k\Lambda^kP_t=P_t^{\otimes k}J_k.
$$

Thus the fermionic matrix element in
{prf:ref}`thm-lqft-record-fock-reconstruction` is exactly

$$
\mathbb E\!\left[
 \overline{J_k(f_1\wedge\cdots\wedge f_k)(S_0^{(1)},\ldots,S_0^{(k)})}
 J_k(g_1\wedge\cdots\wedge g_k)(S_t^{(1)},\ldots,S_t^{(k)})\right],
$$

where the $k$ processes are independent stationary replicas. No independence
of the walkers inside a replica is used. The target is the centered tensor
sector $\mathscr H_k^-$, rather than the whole antisymmetric
$L^2(\pi^{\otimes k})$; for example $1\wedge f$ lies outside that sector.

Under $J=\bigoplus_kJ_k$, the creation and annihilation operators have the
explicit replica formulas

$$
\begin{aligned}
(Ja^\dagger(f)J^{-1}\Psi_k)(s_1,\ldots,s_{k+1})
 &=\frac1{\sqrt{k+1}}\sum_{r=1}^{k+1}(-1)^{r-1}f(s_r)
       \Psi_k(s_1,\ldots,\widehat{s_r},\ldots,s_{k+1}),\\
(Ja(f)J^{-1}\Psi_{k+1})(s_1,\ldots,s_k)
 &=\sqrt{k+1}\int\overline{f(s)}\Psi_{k+1}(s,s_1,\ldots,s_k)d\pi(s).
\end{aligned}
$$

These change the replica sector and use alternating projection or
integration. They specify the operator action beyond scalar multiplication.
:::

:::{prf:proof}
Expand both determinants in the inner product and integrate coordinate by
coordinate. Each product is integrable by Cauchy--Schwarz. With
$G_{ij}=\langle f_i,g_j\rangle$ this gives

$$
\begin{aligned}
\langle J_k(f_1\wedge\cdots\wedge f_k),
        J_k(g_1\wedge\cdots\wedge g_k)\rangle
 &=\frac1{k!}\sum_{\sigma,\tau\in S_k}
   \operatorname{sgn}\sigma\operatorname{sgn}\tau
   \prod_{j=1}^kG_{\sigma(j),\tau(j)}\\
 &=\sum_{\rho\in S_k}\operatorname{sgn}\rho
             \prod_{i=1}^kG_{i,\rho(i)}=\det G.
\end{aligned}
$$

Indeed set $\rho=\tau\sigma^{-1}$; for each $\rho$ there are $k!$
choices of $\sigma$, and
$\operatorname{sgn}\sigma\operatorname{sgn}\tau=\operatorname{sgn}\rho$.
This proves the isometry, including the normalization $1/\sqrt{k!}$.

The permutation operators on the product probability space are unitary.
Their signed average
$\mathcal A_k=(k!)^{-1}\sum_\sigma\operatorname{sgn}\sigma U_\sigma$
is self-adjoint and idempotent: for each product permutation there are
$k!$ pairs in the double sum. Its range is precisely the antisymmetric
subspace. Products of centered functions are dense in $\mathscr H_k$;
applying the bounded projection $\mathcal A_k$ preserves density in its
range. Each projected product is $1/\sqrt{k!}$ times a $J_k$-image.
The isometric image is closed, so this proves surjectivity.

For an orthonormal basis $(e_i)$ of $\mathcal H$, the inverse is explicit:

$$
J_k^{-1}\Psi
=\sum_{i_1<\cdots<i_k}
 \left(\int\overline{J_k(e_{i_1}\wedge\cdots\wedge e_{i_k})(s)}
                         \Psi(s)\,d\pi^{\otimes k}(s)\right)
   e_{i_1}\wedge\cdots\wedge e_{i_k}.
$$

The series converges in sector norm by Parseval's identity, since the
images of the basis wedges are a complete orthonormal family in
$\mathscr H_k^-$. In a finite mode subspace it is a finite sum.

On a product, independence of the transition kernels gives

$$
P_t^{\otimes k}\prod_{j=1}^kf_{\sigma(j)}(s_j)
=\prod_{j=1}^k(P_tf_{\sigma(j)})(s_j).
$$

Summing with permutation signs proves the intertwining on decomposable
wedges; boundedness extends it to the complete sector. Applying the Markov
conditional-expectation identity in the product state space proves the
stated stochastic formula.

Expanding the determinant for $f\wedge g_1\wedge\cdots\wedge g_k$
along its first row gives the creation formula: the normalization ratio
is $\sqrt{k!}/\sqrt{(k+1)!}=1/\sqrt{k+1}$, and its cofactors have signs
$(-1)^{1+r}=(-1)^{r-1}$. To compute its adjoint, integrate each of the
$k+1$ summands against an antisymmetric $\Psi_{k+1}$. Moving $s_r$ to the
first position introduces the same sign, so every term becomes the same
integral; the total factor is $(k+1)/\sqrt{k+1}=\sqrt{k+1}$.
This proves the annihilation formula first on finite products and then by
the bounded operator extensions already proved. $\square$
:::

:::{div} feynman-prose
The alternating insertion formula already tells us how a sign changes when
two modes exchange places. To prove a faithful representation, we must also
check that no further combinations disappear. Apply an ordered insertion
word to the empty replica sector. Different sets of orthonormal modes give
orthogonal vectors, so their coefficients can be recovered from the result.
This is what makes the exterior algebra identification exact.

Keep track of which operation carries that sign. Insertion combines modes
in antisymmetric replica observables. A cloning transition combines
replacement probabilities: applying it to the constant observable still
gives one. The sum of the two composition orders gives two on that
observable, so it cannot vanish. The cloning dynamics enters through the
transition of each complete replica, while the alternating insertion rule specifies its observable
algebra.
:::

:::{prf:theorem} Faithful exterior algebra of alternating recorded insertions
:label: thm-lqft-oriented-word-algebra

Let $E\subset L^2_0(\pi)$ be a finite-dimensional space of the recorded
modes, after quotienting its zero-norm combinations, and let
$e_1,\ldots,e_m$ be an orthonormal basis. On the antisymmetric replica
space of {prf:ref}`thm-lqft-replica-isomorphism`, define
$\mathsf C(f)=Ja^\dagger(f)J^{-1}$ by that theorem's explicit insertion
formula. The algebra generated by these oriented insertions is faithfully
isomorphic to $\Lambda(E)$:

(eq-fg-lq-a1)=
$$
f_1\wedge\cdots\wedge f_k
\longmapsto
\mathsf C(f_1)\cdots\mathsf C(f_k).
\tag{LQ.A1}
$$

Its ordered products $\mathsf C(e_{i_1})\cdots\mathsf C(e_{i_k})$,
$i_1<\cdots<i_k$, together with $I$, are linearly independent.
Consequently its only defining relations are linearity and

(eq-fg-lq-a2)=
$$
\mathsf C(f)\mathsf C(g)=-\mathsf C(g)\mathsf C(f),
\qquad \mathsf C(f)^2=0.
\tag{LQ.A2}
$$

Every linear map $f\mapsto B(f)$ into an associative unital algebra
whose products satisfy these relations factors uniquely through
$\Lambda(E)$. For a different target this factorization can have a
kernel; faithfulness here is supplied by the recorded replica norm.
The adjoints $\mathsf A(f)=\mathsf C(f)^*$ are the already reconstructed
contractions and satisfy

(eq-fg-lq-a3)=
$$
\{\mathsf A(f),\mathsf C(g)\}=\langle f,g\rangle I,
\qquad
\{\mathsf A(f),\mathsf A(g)\}=0.
\tag{LQ.A3}
$$

:::

:::{prf:proof}
**Alternating operation and faithfulness.** The explicit replica
insertion inserts $f$ into each slot with its alternating sign.
Under the already proved unitary $J$, it is exactly left exterior
multiplication. Two insertions therefore reverse sign on exchanging
their order; a repeated insertion vanishes. Equivalently these
relations follow by conjugating the proved creation CAR by $J$.

Linearity first gives a homomorphism from $T(E)$ to the insertion
algebra. Repeated insertions vanish, so the ideal generated by
$f\otimes f$ lies in its kernel. It factors through $\Lambda(E)$,
giving {ref}`(LQ.A1) <eq-fg-lq-a1>`.
To test its remaining kernel, apply an ordered-word linear combination
to the replica vacuum:

$$
\sum_I b_I\mathsf C(e_{i_1})\cdots\mathsf C(e_{i_k})J\Omega
=J\sum_I b_I\,e_{i_1}\wedge\cdots\wedge e_{i_k}.
$$

The Gram-determinant identity gives squared norm $\sum_I|b_I|^2$;
different degrees are orthogonal as well. A zero operator therefore
forces every coefficient to vanish. Anticommutation sorts every word
into one of these ordered words or zero, so they also span. This proves
faithfulness and dimension $2^m$ without assuming that formal
anticommutation alone excludes additional relations.

**Universal property and duals.** For the asserted target map,
$f_1\otimes\cdots\otimes f_k\mapsto B(f_1)\cdots B(f_k)$ is a
homomorphism from $T(E)$. Since $B(f)^2=0$, it annihilates the
defining ideal and has a unique quotient factorization. Its injectivity
requires an argument such as the preceding norm calculation.

Conjugating the explicit adjoint deletion formula by $J$ gives
{ref}`(LQ.A3) <eq-fg-lq-a3>`; its contraction coefficient is the
original $L^2(\pi)$ inner product. For the doubled symbols, apply
the same tensor-quotient construction to $E\oplus E^\vee$.
Polarizing the square of $f+\lambda$ with $f\in E$ and
$\lambda\in E^\vee$ gives $f\lambda+\lambda f=0$.
This proves the mixed exterior relation. If barred multiplication
were identified with $\mathsf A(e_i)$ while unbarred multiplication
were $\mathsf C(e_j)$, that mixed relation at $i=j$ would demand
$I=0$. Thus the two dual constructions have the precisely different
products displayed above.

**Relation to stochastic replacement.** The cloning transition in
{prf:ref}`def-cloning-operator-formal` is a positive probability
kernel $K$ and obeys $K1=1$. For two conservative replacement kernels,

$$
(K_iK_j+K_jK_i)1=2,\qquad K_i^2 1=1.
$$

Their transition composition therefore cannot be the insertion
product in {ref}`(LQ.A2) <eq-fg-lq-a2>`. More generally, if positive
sub-Markov kernels anticommute, each of $K_iK_jf,K_jK_if$ is
nonnegative for $f\ge0$ and their sum is zero. Both compositions
then vanish. The alternating operation represented here is the
oriented insertion on recorded replica sectors. The complete cloning
transition supplies those sectors' evolution through $P_t^{\otimes k}$.
This is the explicit correspondence between the two products.
:::


:::{prf:theorem} Directed-edge coefficients and number-preserving Fock operators
:label: thm-lqft-edge-second-quantization

For finitely many orthonormal modes $e_1,\ldots,e_m$, let $A$ be their
specified directed coefficient matrix. Put $a_i=a(e_i)$ and define

$$
d\Gamma(A)=\sum_{i,j=1}^m A_{ij}a_i^\dagger a_j.
$$

It acts on a wedge by applying $A$ to each occupied factor and summing:

$$
d\Gamma(A)(f_1\wedge\cdots\wedge f_k)
=\sum_{r=1}^kf_1\wedge\cdots\wedge Af_r\wedge\cdots\wedge f_k.
$$

It preserves particle number, restricts to $A$ in the one-particle sector,
and satisfies

$$
[d\Gamma(A),d\Gamma(B)]=d\Gamma([A,B]),\quad
[d\Gamma(A),a^\dagger(f)]=a^\dagger(Af),\quad
\Gamma_-(e^{tA})=e^{t d\Gamma(A)}.
$$

In particular the real edge kernel $\widetilde K=K-K^{\mathsf T}$ yields
a Hermitian matrix $i\widetilde K$ and a Hermitian Fock operator
$d\Gamma(i\widetilde K)$. Its exponential
$e^{-it d\Gamma(i\widetilde K)}$ is unitary. This is the exact finite
Hamiltonian determined by that choice of edge coefficients. Its equality
to the recorded Markov evolution requires equality of their one-particle
operators; skewness of the edge matrix does not assert that equality.
:::

:::{prf:proof}
Applying $a_j$ deletes the $r$th factor with sign $(-1)^{r-1}$ and
coefficient $\langle e_j,f_r\rangle$. Creation inserts $e_i$ in front;
moving it to position $r$ introduces the same sign. Their product is one,
and summing $A_{ij}\langle e_j,f_r\rangle e_i$ gives $Af_r$.
This proves the sector formula and number preservation.

Write $E_{ij}=a_i^\dagger a_j$. The mixed CAR gives

$$
\begin{aligned}
E_{ij}E_{kl}&=\delta_{jk}a_i^\dagger a_l
                  -a_i^\dagger a_k^\dagger a_j a_l,\\
E_{kl}E_{ij}&=\delta_{li}a_k^\dagger a_j
                  -a_k^\dagger a_i^\dagger a_l a_j.
\end{aligned}
$$

The quartic terms agree after two interchanges. Hence
$[E_{ij},E_{kl}]=\delta_{jk}E_{il}-\delta_{li}E_{kj}$.
Multiplying by $A_{ij}B_{kl}$ and summing gives the commutator of matrices.
The same calculation with one creation operator gives
$[E_{ij},a_l^\dagger]=\delta_{jl}a_i^\dagger$ and proves the second identity.
Differentiating
$e^{tA}f_1\wedge\cdots\wedge e^{tA}f_k$ gives the sector formula with
initial value $f_1\wedge\cdots\wedge f_k$; uniqueness of the finite
linear ODE proves the exponential identity on every sector.

Finally $d\Gamma(A)^*=d\Gamma(A^*)$ follows by interchanging $i,j$.
For real skew $\widetilde K$,
$(i\widetilde K)^*=(-i)\widetilde K^{\mathsf T}=i\widetilde K$,
so the Fock generator is Hermitian and its stated exponential is unitary.
The one-particle restriction recovers $A$ exactly; two such lifts can be
equal only when their one-particle operators are equal. $\square$
:::

:::{div} feynman-prose
Think of a fermionic word as a sequence of instructions applied to a list
of modes. Read from right to left: creation inserts a mode, a transition
propagates every mode currently present, and annihilation removes each
possible mode with its alternating sign and inner-product coefficient.
Starting from the vacuum means starting with an empty list. The vacuum
matrix element keeps the terms that return to that empty list.

The numerical input to this calculation is the recorded conditional
expectation kernel. In the two-mode example below, the two possible pairings
give the two terms of a determinant. Independent copies of the complete swarm
realize those products of covariances. This tells us exactly which recorded
statistic the operator word computes.
:::

:::{prf:theorem} Computation of fermionic words from the complete recorded transition
:label: thm-lqft-instantiated-word-evolution

Use the actual conservative stationary transition of
{prf:ref}`def-lqft-record-fock-space`. Its encoded realization is the
explicit kernel in {prf:ref}`thm-sm-instantiated-record-transition`.
Write $C_t=P_t|_{L^2_0(\pi)}$. For every mode $f$,

(eq-fg-lq-r1)=
$$
\begin{aligned}
\Gamma_-(C_t)a^\dagger(f)
 &=a^\dagger(C_tf)\Gamma_-(C_t),\\
a(f)\Gamma_-(C_t)
 &=\Gamma_-(C_t)a(C_t^*f).
\end{aligned}
\tag{LQ.R1}
$$

Together with the CAR, these formulas evaluate every finite word of creation,
annihilation, and recorded transition operators. They retain the actual
whole-swarm dynamics inside every $C_t$.

In continuous time let $\mathbb L$ generate $\Gamma_-(C_t)$. On finite
wedges of generator-domain modes,

(eq-fg-lq-r2)=
$$
\begin{aligned}
[\mathbb L,a^\dagger(f)]\eta&=a^\dagger(Lf)\eta,
 &&f\in\operatorname{Dom}L,\\
[\mathbb L,a(f)]\eta&=-a(L^*f)\eta,
 &&f\in\operatorname{Dom}L^*.
\end{aligned}
\tag{LQ.R2}
$$

If $U$ is the complete-record unitary, the encoded field generator and its
domain are exactly

(eq-fg-lq-r3)=
$$
\widehat{\mathbb L}=\Gamma_-(U)\mathbb L\Gamma_-(U)^{-1},\qquad
\operatorname{Dom}\widehat{\mathbb L}
=\Gamma_-(U)\operatorname{Dom}\mathbb L.
\tag{LQ.R3}
$$

Thus the construction supplies a specified non-Dirac field evolution and
its exact generator identity. Its adjoint and its decay estimates are those
of the recorded contraction semigroup.
:::

:::{prf:proof}
**Insertion and contraction.** Apply the first identity in {ref}`(LQ.R1) <eq-fg-lq-r1>` to
$g_1\wedge\cdots\wedge g_k$. Both sides give
$C_tf\wedge C_tg_1\wedge\cdots\wedge C_tg_k$.
For the second identity, the coefficient of the wedge with the $r$th
factor removed is

$$
(-1)^{r-1}\langle f,C_tg_r\rangle
=(-1)^{r-1}\langle C_t^*f,g_r\rangle
$$

on both sides. Boundedness extends the identities to Fock space.

For a finite word acting on $\Omega$, work from right to left. Creation
inserts a mode, annihilation deletes one with its CAR coefficient, and a
transition applies $C_t$ to each retained mode. After finitely many operations
there is a finite sum of wedges; its vacuum component is the desired matrix
element. Every scalar coefficient is an inner product of modes propagated
by the recorded transitions or their adjoints. Such an inner product is
computed by the conditional-expectation kernel {ref}`(SM.K1) <eq-fg-sm-k1>`, or equivalently by
its original recorded law. For example,

(eq-fg-lq-r4)=
$$
\begin{aligned}
&\langle\Omega,a(f_2)a(f_1)\Gamma_-(C_t)
       a^\dagger(g_1)a^\dagger(g_2)\Omega\rangle\\
&\quad=\langle f_1,C_tg_1\rangle\langle f_2,C_tg_2\rangle
       -\langle f_1,C_tg_2\rangle\langle f_2,C_tg_1\rangle.
\end{aligned}
\tag{LQ.R4}
$$

These products have the independent whole-swarm replica realization in
{prf:ref}`thm-lqft-replica-isomorphism`. A classical four-time moment of a
single swarm is instead the multiplication-and-transition product {ref}`(SM.K4) <eq-fg-sm-k4>`.
Both are determined by the algorithm; the displayed determinant specifies
which statistic represents the fermionic word.

**Differentiation.** The already proved wedge generator differentiates
each factor once. When applied to $f\wedge\eta$, its term differentiating
$f$ is $Lf\wedge\eta$, while its remaining terms are
$f\wedge\mathbb L\eta$. Subtraction proves the first commutator.
For annihilation, differentiation of the deleted factor contributes
$-\langle f,Lg_r\rangle=-\langle L^*f,g_r\rangle$ in the difference
$\mathbb L a(f)-a(f)\mathbb L$. This is the second commutator, with the
same deletion signs. All other differentiated factors cancel.

**Encoding.** On a wedge, {ref}`(SM.K2) <eq-fg-sm-k2>` gives

$$
\Gamma_-(\widehat C_t)\Gamma_-(U)
=\Gamma_-(U)\Gamma_-(C_t).
$$

Differentiate in the Fock norm. Since $\Gamma_-(U)$ is a unitary onto,
existence of a derivative on either side is equivalent to existence on the
other. This proves the full domain equality in {ref}`(LQ.R3) <eq-fg-lq-r3>`. It also gives the
resolvent identity and transports every finite bounded-operator matrix
element. The discrete-time formulas use powers of the actual kernel and
require no infinitesimal generator.
:::

:::{div} feynman-prose
So far the recorded transition has acted on Fock vectors. We can also use
it to evolve the operators we measure. There is one detail to get right:
propagating a mode can reduce its norm, so substituting propagated modes
into every factor of an arbitrary product would change the scalar term in
the CAR.

The prescription below first moves creation operators to the left using
the CAR, retaining every scalar contraction, and then propagates the modes
in those normally ordered terms. The calculated multiplicative defect
measures the correction to simple factor-by-factor substitution. The result
preserves the identity and is completely positive: it preserves positivity
for every positive matrix of operators. The proof realizes this using an
extra Hilbert-space summand that accounts for the contraction's
norm deficit. It leaves the recorded update and its time parameter intact;
reversibility is not needed for this construction.

The link to the previous calculation is exact. Acting on the vacuum, an
evolved operator gives the same vector as propagating its original vacuum
vector. Repeating this identity reproduces every ordered nested correlation
in the theorem from the earlier Fock transition words.
:::

:::{prf:theorem} Completely positive CAR evolution of the recorded contraction
:label: thm-lqft-record-car-channel

Use the conservative stationary recorded process and centered mode space
$\mathcal H=L^2_0(\pi)$ of
{prf:ref}`thm-lqft-instantiated-word-evolution`. Let
$\mathfrak A_{\mathrm{CAR}}(\mathcal H)$ be the norm-closed unital algebra
generated by its bounded creation and annihilation operators. The actual
contractions $C_t=P_t|_{\mathcal H}$ determine a unique unital completely
positive map $\mathcal Q_t$ with the following action on normally ordered
words:

(eq-fg-lq-c1)=
$$
\mathcal Q_t\!\left(
 a^\dagger(f_1)\cdots a^\dagger(f_p)a(g_q)\cdots a(g_1)\right)
=a^\dagger(C_tf_1)\cdots a^\dagger(C_tf_p)
 a(C_tg_q)\cdots a(C_tg_1).
\tag{LQ.C1}
$$

The empty word maps to $I$. These maps preserve the Fock vacuum state
$\omega$, obey $\mathcal Q_t\mathcal Q_s=\mathcal Q_{t+s}$, and are
pointwise norm continuous when the recorded semigroup is strongly
continuous. Their exact recorded two-point correspondence is

(eq-fg-lq-c2)=
$$
\omega\!\left(a(f)\mathcal Q_t(a^\dagger(g))\right)
=\langle f,C_tg\rangle
=\mathbb E_\pi[\overline{f(S_0)}g(S_t)].
\tag{LQ.C2}
$$

Its action on vacuum-generated vectors is the already constructed Fock
contraction. In particular, for $A_j\in\mathfrak A_{\mathrm{CAR}}$ and
nonnegative time increments $t_1,\ldots,t_n$,

(eq-fg-lq-c4)=
$$
\begin{aligned}
\mathcal Q_t(A)\Omega&=\Gamma_-(C_t)A\Omega,\\
\omega\!\left(A_0\mathcal Q_{t_1}
 \left(A_1\mathcal Q_{t_2}(\cdots\mathcal Q_{t_n}(A_n)\cdots)\right)\right)
&=\langle\Omega,A_0\Gamma_-(C_{t_1})A_1
          \cdots\Gamma_-(C_{t_n})A_n\Omega\rangle.
\end{aligned}
\tag{LQ.C4}
$$

Thus the complete positive algebra evolution and the previous Fock word
calculation have exactly the same ordered regression matrix elements.
In discrete time use $C_k=C_1^k$ and the corresponding powers
of $\mathcal Q_1$. Encoding the complete record intertwines these maps
through the CAR isomorphism induced by its unitary $U$.

The multiplicative defect is calculated by

(eq-fg-lq-c3)=
$$
\mathcal Q_t(a(f)a^\dagger(g))
 -\mathcal Q_t(a(f))\mathcal Q_t(a^\dagger(g))
=\langle f,(I-C_t^*C_t)g\rangle I.
\tag{LQ.C3}
$$

Thus this completely positive evolution is multiplicative exactly when
$C_t$ is an isometry. For a unitary $C_t$ it is a CAR automorphism.
The recorded dissipative dynamics and its closed-system unitary special
case are distinguished by this same calculated defect.
:::

:::{prf:proof}
**An isometric realization from the recorded kernel.** Conditional Jensen
and stationarity already give $\|C_t\|\le1$. Therefore
$D_t=(I-C_tC_t^*)^{1/2}$ exists, and

$$
J_t:\mathcal H\longrightarrow\mathcal H\oplus\mathcal H,\qquad
J_tf=(C_t^*f,D_tf)
$$

is an isometry:
$J_t^*J_t=C_tC_t^*+D_t^2=I$. Its exterior lift
$V_t=\Gamma_-(J_t)$ is consequently an isometry. Let $\iota$ denote
the canonical CAR embedding into the first summand, defined by
$\iota(a^\dagger(f))=a^\dagger(f\oplus0)$. Define

$$
\mathcal Q_t(A)=V_t^*\iota(A)V_t.
$$

For a positive matrix $[A_{ij}]$ of algebra elements and Fock vectors
$\xi_1,\ldots,\xi_m$,

$$
\sum_{i,j}\langle\xi_i,\mathcal Q_t(A_{ij})\xi_j\rangle
=\sum_{i,j}\langle V_t\xi_i,\iota(A_{ij})V_t\xi_j\rangle\ge0.
$$

This proves complete positivity at every matrix size. The isometry gives
$\mathcal Q_t(I)=I$ and $\|\mathcal Q_t(A)\|\le\|A\|$.
The additional Hilbert summand realizes this map mathematically; it
introduces no change to the recorded swarm transition.

**The full word formula.** The insertion and contraction proof of
{ref}`(LQ.R1) <eq-fg-lq-r1>` also applies between two mode spaces. Since
$J_t^*(f\oplus0)=C_tf$, it gives

$$
a(f\oplus0)V_t=V_ta(C_tf),\qquad
V_t^*a^\dagger(f\oplus0)=a^\dagger(C_tf)V_t^*.
$$

Move all annihilators through $V_t$ from the right and all creators
through $V_t^*$ from the left. The remaining factor is $V_t^*V_t=I$,
proving {ref}`(LQ.C1) <eq-fg-lq-c1>`. Every finite CAR polynomial can be
normally ordered by repeatedly using the CAR. Normally ordered
polynomials are therefore norm dense in $\mathfrak A_{\mathrm{CAR}}$.
The formula places their images in that same algebra, and the contraction
bound extends this statement to its closure. Density also proves
uniqueness.

**Semigroup, state, and continuity.** On each normally ordered word,
successive application replaces every mode by $C_tC_sf=C_{t+s}f$.
Density and contraction give the semigroup identity on the full algebra.
The vacuum maps under $V_t$ to the larger Fock vacuum. Hence
$\omega(\mathcal Q_t(A))=\omega(A)$, first on polynomials and then by
continuity. The two-point equality follows from
$a^\dagger(C_tg)\Omega=C_tg$ and the existing conditional-expectation
identity.

For a fixed word, telescope the difference between its transformed
factors at $t$ and at zero. The equality $\|a(f)\|=\|f\|$ and strong
continuity of $C_t$ make each term tend to zero in operator norm.
Approximation by polynomials and the uniform contraction bound prove
pointwise norm continuity for arbitrary $A$. A finite nested regression
expression is evaluated by normal ordering after each insertion and
applying {ref}`(LQ.C1) <eq-fg-lq-c1>`; all contractions are the recorded
inner products already evaluated in {ref}`(LQ.R4) <eq-fg-lq-r4>`.
For a normally ordered word containing an annihilator, both sides of
$\mathcal Q_t(A)\Omega=\Gamma_-(C_t)A\Omega$ vanish. For a word
containing only creators, both are the wedge of its $C_t$-propagated
modes. Normal ordering and norm density prove the first identity in
{ref}`(LQ.C4) <eq-fg-lq-c4>` for every $A$ in the algebra. Apply it
recursively to the nested expression to prove the second identity.
This instantiates the entire ordered Fock-word correspondence as a
completely positive algebra evolution. A single-swarm multiplication
correlation retains its separately identified formula
{ref}`(SM.K4) <eq-fg-sm-k4>`.

**Multiplication and encoding.** The CAR gives
$a(f)a^\dagger(g)=\langle f,g\rangle I-a^\dagger(g)a(f)$.
Applying {ref}`(LQ.C1) <eq-fg-lq-c1>` and subtracting the product of
the images proves {ref}`(LQ.C3) <eq-fg-lq-c3>`. Multiplicativity
therefore forces $C_t^*C_t=I$. Conversely an isometry preserves the CAR,
so substitution of its modes defines a $*$-homomorphism; normal ordering
identifies it with $\mathcal Q_t$. A unitary has the inverse substitution.

Finally $\widehat C_tU=UC_t$. On every normally ordered word the CAR
isomorphism $\alpha_U(a^\dagger(f))=a^\dagger(Uf)$ therefore satisfies
$\widehat{\mathcal Q}_t\alpha_U=\alpha_U\mathcal Q_t$.
Norm density extends the identity. This constructs an algebra evolution
for the same native two-point data in addition to the Fock-vector
contraction $\Gamma_-(C_t)$.
:::


:::{prf:corollary} Existing LSI and native contraction bounds
:label: cor-lqft-fock-lsi-transfer

For centered modes in the pulled-back Sobolev domain of
{prf:ref}`thm-sm-direct-existing-machinery`, its previously proved LSI
constant gives

$$
\|a(f)\|^2=\|f\|_{L^2(\pi)}^2
\le C_*\mathcal E(f).
$$

Under the stationary contraction bound of
{prf:ref}`thm-cluster-decomposition`, on the $k$th centered sector,

$$
\|\Lambda^kP_t\|\le\min\{1,M^ke^{-k\lambda t}\}.
$$

:::

:::{prf:proof}
The LSI-to-Poincare calculation in
{prf:ref}`thm-sm-direct-existing-machinery` gives
$\|f\|_2^2=\operatorname{Var}_\pi f\le C_*\mathcal E(f)$ for real
centered $f$. Apply it to real and imaginary parts and add for complex
$f$. The CAR norm identity proves the first assertion.
Taking the $k$fold tensor norm of the centered semigroup proves the second;
its independent contraction bound supplies the minimum with one.

$\square$
:::

:::{div} feynman-prose
Choose the recorded values that describe a region, then form centered
measurements from those values. To compare two regions, calculate the
covariance of their measurements under the recorded law. That number is
exactly the coefficient of their mixed CAR anticommutator. Repeating the
calculation at two observation times measures how the actual update carries
dependence between the regions.

For a finite list of measurements, these calculations produce ordinary
Gram and transition matrices. The Gram matrix identifies redundant modes
and normalizes the independent ones. Its cross-region entries determine
the operator relations, and the bound below carries those same entries
into products of several operators. Every quantity comes from the chosen
recorded measurements and their joint law.
:::

:::{prf:theorem} Local record algebras and their computed locality defect
:label: thm-lqft-record-locality-defect

Use the actual stationary record law and centered mode space of
{prf:ref}`def-lqft-record-fock-space`. A recorded region $O$ specifies a
measurable descriptor $q_O$ of the complete state. Define
$\mathcal H_O=L^2_0(\sigma(q_O),\pi)\subset\mathcal H$ and let
$\mathfrak A(O)$ be the CAR algebra generated by modes in $\mathcal H_O$.
The orthogonal projection onto this closed subspace is
$\Pi_Of=\mathbb E_\pi[f\mid\sigma(q_O)]$ for centered $f$.
For nested descriptors these algebras are isotone. Their locality and their
propagation under the *recorded* kernel are determined exactly by

(eq-fg-lq-s1)=
$$
\begin{aligned}
\{a(f),a^\dagger(g)\}
 &=\operatorname{Cov}_\pi(\overline f,g)I,
 &f\in\mathcal H_O,\quad g\in\mathcal H_V,\\
\{a(f),\mathcal Q_t(a^\dagger(g))\}
 &=\mathbb E_\pi[\overline{f(S_0)}g(S_t)]I
 =\langle f,\Pi_OC_t\Pi_Vg\rangle I.
\end{aligned}
\tag{LQ.S1}
$$

Here $\operatorname{Cov}_\pi(\overline f,g)$ means
$\pi(\overline f g)-\pi(\overline f)\pi(g)$, and $C_t$ and $\mathcal Q_t$
are the already constructed recorded contractions and CAR channels.
The norm of the block $\Pi_OC_t\Pi_V$ is precisely the largest absolute
mixed anticommutator coefficient over unit modes in the two regions.
At equal time the algebras are graded commuting exactly when
$\mathcal H_O\perp\mathcal H_V$. In that case their even parts commute.

For finite mode lists $f_1,\ldots,f_r$ and $g_1,\ldots,g_s$, all these
coefficients are entries of the recorded matrices

(eq-fg-lq-s2)=
$$
G_{ij}=\mathbb E_\pi[\overline{f_i(S_0)}f_j(S_0)],
\qquad
K_{ij}(t)=\mathbb E_\pi[\overline{f_i(S_0)}g_j(S_t)].
\tag{LQ.S2}
$$

Thus neither spatial separation of the descriptor labels nor genealogy
alone substitutes for the computed orthogonality condition.

*Proof.* Conditional expectation is the orthogonal projection because,
for every centered $\sigma(q_O)$-measurable $u$,
$\langle u,f-\mathbb E[f\mid\sigma(q_O)]\rangle=0$.
Measurability inclusion gives inclusion of the mode subspaces and hence of
the generated algebras. The creation and annihilation calculation in
{prf:ref}`thm-lqft-record-fock-reconstruction` gives the first line of
{ref}`(LQ.S1) <eq-fg-lq-s1>`. The normally ordered channel formula in
{prf:ref}`thm-lqft-record-car-channel` gives
$\mathcal Q_t(a^\dagger(g))=a^\dagger(C_tg)$, proving the second.
Stationarity and conditional expectation identify its coefficient with
the displayed actual two-time expectation. Taking the supremum over unit
$f,g$ is the definition of the norm of the cross block.

For clarity, let $X=x_1\cdots x_p$ and $Y=y_1\cdots y_q$ be words of
creation or annihilation operators with respective modes $f_i,g_j$.
Successively interchange each $x_i$ with each $y_j$. Each interchange
contributes its minus sign and possibly one scalar contraction. Therefore

(eq-fg-lq-s3)=
$$
\|XY-(-1)^{pq}YX\|
\le \sum_{i=1}^p\sum_{j=1}^q
 |\langle f_i,g_j\rangle|
 \prod_{\ell\ne i}\|f_\ell\|
 \prod_{k\ne j}\|g_k\|.
\tag{LQ.S3}
$$

Terms of equal creation/annihilation type have zero contraction and may
be omitted from this upper bound. This follows also by induction on the
number of interchanges, using the operator norm identity
$\|a(f)\|=\|f\|$. Orthogonality makes every term zero; even degrees give
ordinary commutation. Extend from words to the norm-closed even and odd
subspaces. Conversely graded commutation applied to the two odd generators
$a(f),a^\dagger(g)$ forces $\langle f,g\rangle=0$.

The finite Gram matrix is positive semidefinite since
$c^*Gc=\|\sum_i c_if_i\|^2$. Diagonalize its nonzero part as
$G=V\Lambda V^*$, with $\Lambda>0$, and put
$e_\alpha=\sum_i f_i(V\Lambda^{-1/2})_{i\alpha}$.
Direct multiplication gives
$\langle e_\alpha,e_\beta\rangle=\delta_{\alpha\beta}$.
These give canonical finite CAR modes after quotienting the null modes.
If this change of basis combines descriptors in different regions, the
new modes have that combined dependence. Orthogonalizing the Gram matrix
is an exact representation calculation, not a proof that the original
geometric regions were local. $\square$
:::

:::{prf:theorem} Exact inherited-history covariance of native regional readouts
:label: thm-lqft-inherited-history-covariance

Fix the existing finite execution law of
{prf:ref}`thm-sm-instantiated-record-transition`, including its prescribed
survival conditioning when present. Denote expectation under this same law
by $\mathbb E$. Let $F,G$ be bounded cylinders of the original regional
readouts, retaining their masks and normalizations. Let $\mathcal F_j$
be the history through the $j$th recorded update, with $\mathcal F_0$
containing the initial record and $\mathcal F_T$ the complete record used
by these cylinders. Write

$$
\begin{gathered}
f=F-\mathbb EF,\qquad g=G-\mathbb EG,\qquad
m_j^F=\mathbb E[f\mid\mathcal F_j],\quad
m_j^G=\mathbb E[g\mid\mathcal F_j],\\
d_j^F=m_j^F-m_{j-1}^F,\qquad d_j^G=m_j^G-m_{j-1}^G,\qquad
B_k=\mathbb E[\overline{m_k^F}m_k^G].
\end{gathered}
$$

Thus $B_k$ is precisely the covariance of the two conditional means,
with the first argument conjugated. No change of regional modes is made.
Its exact value and an update-resolved bound are

$$
\begin{aligned}
B_k&=B_0+\sum_{j=1}^k
             \mathbb E[\overline{d_j^F}d_j^G],\\
|B_k|&\le |B_0|+\sum_{j=1}^k\sqrt{e_j^F e_j^G},
\qquad e_j^F=\mathbb E|d_j^F|^2,\quad
e_j^G=\mathbb E|d_j^G|^2.
\end{aligned}
$$

Define complex variance by $\operatorname{Var}(F)=\mathbb E|f|^2$ and
conditional variance by the corresponding conditional squared modulus.
The bound retaining the total predictable variance is

$$
\begin{gathered}
v_k^F:=\operatorname{Var}(F)
       -\mathbb E\operatorname{Var}(F\mid\mathcal F_k)
       =\|m_0^F\|_2^2+\sum_{j=1}^k e_j^F,\\
|B_k|\le\sqrt{v_k^Fv_k^G}
       \le\sqrt{\operatorname{Var}(F)\operatorname{Var}(G)}.
\end{gathered}
$$

For positive variances, set
$\eta_k^F=v_k^F/\operatorname{Var}(F)$ and similarly for $G$. Then
$0\le\eta_k^F,\eta_k^G\le1$ and

$$
\frac{|B_k|}{\sqrt{\operatorname{Var}(F)\operatorname{Var}(G)}}
\le\sqrt{\eta_k^F\eta_k^G}.
$$

If either variance is zero, $B_k=0$. For the full cross coefficient the
same decomposition gives the exact remainder

$$
\mathbb E[\overline f g]-B_k
=\sum_{j=k+1}^T\mathbb E[\overline{d_j^F}d_j^G]
=\mathbb E\operatorname{Cov}(\overline F,G\mid\mathcal F_k).
$$

**Evaluation with the recorded update.** Let $R_j$ denote the already
retained history through update $j$, and let
$K_j(r,dr')$ be its conditional next-history kernel under the selected
execution law. These are the original kernels acting on retained histories.
Starting from $m_T^F(r)=F(r)-\mathbb EF$, backward integration gives

$$
\begin{aligned}
m_{j-1}^F(r)&=\int m_j^F(r')K_j(r,dr'),\\
b_j(r)&=\int
 \overline{\bigl(m_j^F(r')-m_{j-1}^F(r)\bigr)}
 \bigl(m_j^G(r')-m_{j-1}^G(r)\bigr)K_j(r,dr'),\\
B_k&=\int\overline{m_0^F(r)}m_0^G(r)\,\mathcal L(R_0)(dr)
     +\sum_{j=1}^k\int b_j(r)\,\mathcal L(R_{j-1})(dr).
\end{aligned}
$$

Replacing the mixed integrand in $b_j$ by its first squared modulus
gives $e_j^F$ after integration over $R_{j-1}$, and likewise for $G$.
For an unselected update, the kernel integral is exactly integration of
the recorded update map against its fresh-input law in
{prf:ref}`def-sm-complete-update-law`. For the specified latent kinetic
stages this map includes squashing before both A transports, as in
{prf:ref}`cor-ym-squashed-kinetic-support`.

For a history conditioned to survive through $T$, the conditional kernel
is instead the already specified survival-weighted kernel of
{prf:ref}`prop-ym-qsd-history-identification`. Explicitly, if
$Q_j(r,dr')$ is the killed history-extension kernel, put
$h_T=1$ and $h_{j-1}=Q_jh_j$. On histories of positive selected
probability,

$$
K_j(r,dr')=\frac{Q_j(r,dr')h_j(r')}{h_{j-1}(r)}.
$$

Its initial law is weighted by $h_0$ and normalized by the original
survival probability. Thus the covariance calculation includes the same
selection in both its backward integrals and its outer expectations.

For terminal state readouts $F=A(S_{k+r})$, $G=B(S_{k+s})$ under
the existing conservative stationary law $\pi$, the formula specializes to

$$
B_k=\langle C_r a,C_s b\rangle_\pi
   =\langle a,C_r^*C_s b\rangle_\pi,\qquad
a=A-\pi A,\quad b=B-\pi B.
$$

Here $C_r$ is the centered native $r$-update contraction. The inherited term uses
$C_r^*C_s$, whereas the unequal-time cross coefficient in
{prf:ref}`thm-lqft-record-locality-defect` uses the recorded time lag.

*Proof.* Conditional expectation makes $m_j^F,m_j^G$ square-integrable
martingales. For $i<j$, the tower identity gives

$$
\mathbb E[\overline{d_i^F}d_j^G]
=\mathbb E[\overline{d_i^F}
                 \mathbb E(d_j^G\mid\mathcal F_{j-1})]=0.
$$

The same argument handles $i>j$ and the cross terms with the initial
conditional means. Expanding the two martingale sums proves the formula
for $B_k$; Cauchy--Schwarz bounds each term. Applying this orthogonality
to a single readout proves the sum for $v_k^F$. The conditional mean and
its residual are orthogonal, so their squared norms sum to
$\operatorname{Var}(F)$. This gives its other expression and
$0\le v_k^F\le\operatorname{Var}(F)$. Cauchy--Schwarz applied directly
to $m_k^F,m_k^G$ proves the predictable-variance bound. Since
$m_T^F=f$ and $m_T^G=g$, subtracting the two martingale expansions
gives the remainder; expanding the two conditional residuals identifies
it with the conditional covariance.

The backward formula is the tower identity for the actual history
extension. Its difference is $d_j^F$, so integration of its mixed
product proves the displayed expression for $b_j$ and $B_k$.
Bayes' rule gives the survival-weighted kernel and initial law; the
recursion for $h_j$ verifies that each such kernel has mass one.
Finally, the Markov property gives
$\mathbb E[a(S_{k+r})\mid\mathcal F_k]=C_ra(S_k)$.
Stationarity and the definition of the adjoint prove the last identity.
All these steps use the same execution law as the two readouts. $\square$
:::

:::{prf:corollary} Full covariance bound from the recorded history increments
:label: cor-lqft-full-history-locality-bound

Use the readouts and the same law of
{prf:ref}`thm-lqft-inherited-history-covariance`. For positive variances,
write $\sigma_F^2=\operatorname{Var}(F)$,
$\sigma_G^2=\operatorname{Var}(G)$, and set

$$
\begin{gathered}
\alpha_0=\|m_0^F\|_2^2/\sigma_F^2,\quad
\alpha_j=e_j^F/\sigma_F^2\quad(1\le j\le T),\\
\beta_0=\|m_0^G\|_2^2/\sigma_G^2,\quad
\beta_j=e_j^G/\sigma_G^2\quad(1\le j\le T),\qquad
\varepsilon_{FG}=\sum_{j=0}^T\sqrt{\alpha_j\beta_j}.
\end{gathered}
$$

These numbers are computed by the native-kernel recursion in that theorem.
They obey $\sum_j\alpha_j=\sum_j\beta_j=1$ and
$0\le\varepsilon_{FG}\le1$. In particular, $\varepsilon_{FG}=0$
exactly when $\alpha_j\beta_j=0$ for every $j$, including $j=0$.
The exact signed covariance can also vanish by cancellation when this
nonnegative upper bound is positive. For the original normalized centered modes,
the complete cross coefficient satisfies

$$
\begin{aligned}
s_{FG}
&=\frac{B_k+
  \mathbb E\operatorname{Cov}(\overline F,G\mid\mathcal F_k)}
 {\sigma_F\sigma_G}
 =\frac{B_0+\sum_{j=1}^T\mathbb E[\overline{d_j^F}d_j^G]}
 {\sigma_F\sigma_G},\\
|s_{FG}|&\le\varepsilon_{FG}
\le\sqrt{\eta_k^F\eta_k^G}
 +\sqrt{(1-\eta_k^F)(1-\eta_k^G)}\le1
\qquad(0\le k\le T).
\end{aligned}
$$

For terminal-state readouts under the conservative stationary law used
by the existing CAR representation, apply this calculation to their
stationary recorded histories. Their terminal marginal is that same law.
For its unit modes
$f=(F-\mathbb EF)/\sigma_F$, $g=(G-\mathbb EG)/\sigma_G$, substitution
in the exact even-observable relation gives

$$
[n_f,n_g]=s_{FG}a^\dagger(f)a(g)
             -\overline{s_{FG}}a^\dagger(g)a(f),\qquad
\|[n_f,n_g]\|\le b(\varepsilon_{FG}),
$$

where

$$
b(\varepsilon)=
\begin{cases}
\varepsilon\sqrt{1-\varepsilon^2},&0\le\varepsilon\le1/\sqrt2,\\
1/2,&1/\sqrt2\le\varepsilon\le1.
\end{cases}
$$

The bound also holds on the reducing observable vacuum sector of
{prf:ref}`thm-ym-hk-record-instantiation`. For arbitrary even words,
replace each cross inner product in its word estimate by
$\varepsilon_{F_iG_j}\|f_i\|\|g_j\|$, with the corresponding normalized
readouts used to compute $\varepsilon_{F_iG_j}$.

*Proof.* The terminal predictable variance is the full variance, proving
the two sum identities. The exact covariance decomposition and
Cauchy--Schwarz on each increment give the first bound. Group the indices
$0,\ldots,k$ and $k+1,\ldots,T$ and apply Cauchy--Schwarz to each group.
Their respective sums are $\eta_k^F,1-\eta_k^F$ and
$\eta_k^G,1-\eta_k^G$. A final two-dimensional Cauchy--Schwarz inequality
gives the upper bound one. The even CAR identity is
{prf:ref}`thm-ym-hk-record-instantiation`; on full Fock space its norm is
$|s_{FG}|\sqrt{1-|s_{FG}|^2}$. Maximizing this expression over
$0\le|s_{FG}|\le\varepsilon_{FG}$ gives the stated piecewise function.
Restriction to a reducing subspace cannot increase the norm. Substitution
in the established even-word estimate proves the last statement. $\square$
:::

:::{div} feynman-prose
Take a mode and calculate its conditional expected value after one complete
algorithm step. This is the mode $Pf$ that enters the next CAR operator.
For a normally ordered word, apply the same recorded transition to every
mode. Selection, cloning, and kinetics enter through that complete update.
In the established continuous-time model, differentiating this calculation
gives the displayed sum of generator terms.

We can also calculate how much of the evolved mode a chosen regional
descriptor can express. Conditional expectation onto that descriptor gives
the best approximation in the recorded norm; the residual is exactly the
operator approximation error below. Encoding the complete record preserves
these projections, transition coefficients, and errors. This makes both
evolution and localization computable in either recorded representation.
:::

:::{prf:theorem} Native fermion evolution, localization, and record covariance
:label: thm-lqft-native-local-fermion-evolution

For the recorded semigroup of
{prf:ref}`thm-sm-instantiated-record-transition`, its generator $L$ on
centered modes determines the CAR-channel generator on normally ordered
words whose modes lie in $D(L)$:

(eq-fg-lq-s6)=
$$
\begin{aligned}
\mathscr L_{\mathrm{CAR}}
 \big[a^\dagger(f_1)\cdots a^\dagger(f_p)a(g_q)\cdots a(g_1)\big]
={}&\sum_{i=1}^p
 a^\dagger(f_1)\cdots a^\dagger(Lf_i)\cdots
 a^\dagger(f_p)a(g_q)\cdots a(g_1)\\
&+\sum_{j=1}^q
 a^\dagger(f_1)\cdots a^\dagger(f_p)
 a(g_q)\cdots a(Lg_j)\cdots a(g_1).
\end{aligned}
\tag{LQ.S6}
$$

For the implemented discrete update the exact statement is instead the
normally ordered substitution $f_i\mapsto Pf_i$, $g_j\mapsto Pg_j$.
All selection, cloning and kinetic terms enter through this complete $P$
or its established continuous-time generator $L$.
For a mode initially in $\mathcal H_O$, its distance after time $t$ from
the modes of a target descriptor $V$ is exactly

(eq-fg-lq-s7)=
$$
\inf_{u\in\mathcal H_V}
 \|\mathcal Q_t(a^\dagger(f))-a^\dagger(u)\|
 =\|(I-\Pi_V)C_tf\|.
\tag{LQ.S7}
$$

The full recorded encoding unitary $U$ transports these algebras,
localization errors and time correlations exactly: with
$\widehat{\mathcal H}_O=U\mathcal H_O$ and
$\widehat C_t=UC_tU^{-1}$, the induced CAR isomorphism obeys

(eq-fg-lq-s8)=
$$
\alpha_U(a^\dagger(f))=a^\dagger(Uf),\qquad
\alpha_U\mathcal Q_t=\widehat{\mathcal Q}_t\alpha_U,
\qquad
\widehat\Pi_O=U\Pi_OU^{-1}.
\tag{LQ.S8}
$$

These identities establish covariance under the proved record
identification. They do not identify a geometric transformation absent
from that identification.

*Proof.* For $f\in D(L)$, the semigroup definition gives
$\|t^{-1}(C_tf-f)-Lf\|\to0$.
The norm identity for creation and annihilation transfers this convergence
to their bounded operators. Apply the finite product difference identity
to the normally ordered formula of
{prf:ref}`thm-lqft-record-car-channel`. Every unchanged factor stays bounded
as $t\downarrow0$, so the terms with one difference quotient converge to
{ref}`(LQ.S6) <eq-fg-lq-s6>`. No product rule on arbitrary unordered words
is asserted: contractions must first be retained using the CAR. In
particular the multiplicative defect already calculated in that theorem
remains present.

The same norm identity makes the infimum in
{ref}`(LQ.S7) <eq-fg-lq-s7>` equal to
$\inf_{u\in\mathcal H_V}\|C_tf-u\|$; orthogonal projection attains it.
For a local mode $f\in D(L)\cap\mathcal H_V$, it also yields the explicit
short-time calculation
$\|(I-\Pi_V)C_tf-t(I-\Pi_V)Lf\|=o(t)$.
Thus the cross component of the *actual* generator, or the exact
one-step component $(I-\Pi_V)Pf$, determines leakage from the chosen
region. These are the same conditional-expectation blocks used in
{prf:ref}`thm-sm-direct-channel-memory`; discarded dependencies can be
retained through its exact memory formula.

Unitary preservation of the inner product preserves all CAR relations.
Conjugation by $\Gamma_-(U)$ implements $\alpha_U$ on the represented
algebras. On normally ordered words the identity $UC_t=\widehat C_tU$
proves the intertwining in {ref}`(LQ.S8) <eq-fg-lq-s8>`; norm continuity
extends it to the full algebra. Unitary transport of orthogonal projections
proves its last identity, and hence preserves every norm and scalar in
{ref}`(LQ.S1) <eq-fg-lq-s1>` and
{ref}`(LQ.S7) <eq-fg-lq-s7>`.

The Fractal Set causal order in
{prf:ref}`thm-fractal-is-causal-set` and its finite reconstruction in
{prf:ref}`prop-fractal-cst-framework-lift` assign recorded regions and
ancestry, so they provide concrete descriptors for this theorem. The
ancestral intervention identity of {prf:ref}`lem-no-signaling-fg` states
which recorded updates are unchanged when their inputs are unchanged.
It does not set the cross-covariance in
{ref}`(LQ.S1) <eq-fg-lq-s1>` to zero: two regions can share random
ancestors. Consequently a relativistic local-net identification must specify
which of these actual mode spaces correspond to its spacelike regions
and prove the corresponding zero cross blocks. The geometric consistency
estimator and its derivative estimates do not change the mode inner
product. This is an explicit compatibility calculation, not an extra
continuum assumption.

$\square$
:::


:::{prf:proposition} Product obstruction and the distinction from same-swarm moments
:label: thm-lqft-product-obstruction

For nonzero $\mathcal H$, the unital commutative algebra of bounded scalar
record observables is not algebra-isomorphic to the full represented CAR
algebra. The map $f\mapsto a^\dagger(f)$ is linear; it is not a
multiplicative identification of numerical readouts with creation operators.
The isomorphism in {prf:ref}`thm-lqft-replica-isomorphism` identifies the
antisymmetric replica sectors and their dynamics. It does not identify
arbitrary moments of observables within one interacting swarm with the
fermionic determinant formula.
:::

:::{prf:proof}
An algebra homomorphism from a commutative algebra has commuting image,
since $\Phi(f)\Phi(g)=\Phi(fg)=\Phi(gf)=\Phi(g)\Phi(f)$.
For a unit mode let $a=a(f)$. If $a$ and $a^\dagger$ commuted, their mixed
CAR would give $aa^\dagger=\tfrac12 I$. But $a^2=0$; multiplying the
previous equation on the left by $a$ gives $0=\tfrac12 a$, contradicting
$aa^\dagger=\tfrac12 I$. Thus the full CAR algebra is noncommutative.

The distinction between moments already appears for two equal centered
unit modes $f=g$. The two-particle exterior vector $f\wedge f$ vanishes,
and its equal-time determinant is

$$
\det\begin{pmatrix}1&1\\1&1\end{pmatrix}=0.
$$

If $f$ is a bounded nonzero centered record observable, its same-record
fourth moment instead satisfies
$\mathbb E|f|^4\ge(\mathbb E|f|^2)^2=1$.
The determinant therefore cannot equal that ordinary fourth moment.
In a nontrivial record probability space a concrete bounded centered mode
is $(\mathbf1_A-\pi(A))/\sqrt{\pi(A)(1-\pi(A))}$ for
$0<\pi(A)<1$. This argument applies without changing any transition rule.
It establishes the necessity of the antisymmetrization in the exact
replica correspondence. $\square$
:::

:::{prf:theorem} Clifford identities of recorded CAR modes
:label: thm-dirac-structure-lqft

For orthonormal recorded modes $e_1,\ldots,e_m$ in the existing mode space,
write $a_j=a(e_j)$ and define

$$
\Gamma^{2j-1}=a_j+a_j^\dagger,\qquad
\Gamma^{2j}=i(a_j^\dagger-a_j).
$$

Then the $\Gamma^a$ are Hermitian and
$\{\Gamma^a,\Gamma^b\}=2\delta^{ab}I$.
For four such operators define $\gamma^0=i\Gamma^1$ and
$\gamma^r=\Gamma^{r+1}$ for $1\le r\le3$. Their anticommutators are
$2\eta^{\mu\nu}I$, with $\eta=\operatorname{diag}(-1,1,1,1)$.

*Proof.* Expand each anticommutator with the CAR of
{prf:ref}`thm-lqft-record-fock-reconstruction`. The equal-index terms give
$2I$ and all mixed terms cancel. Multiplying the first operator by $i$
changes its square to $-I$. These are identities of the same recorded
operators. $\square$
:::

:::{prf:definition} Chiral projectors of the recorded Clifford operators
:label: def-lqft-chiral-projectors

For the four operators in {prf:ref}`thm-dirac-structure-lqft`, set

$$
\gamma^5=i\gamma^0\gamma^1\gamma^2\gamma^3,\qquad
P_L=\frac{I-\gamma^5}{2},\qquad P_R=\frac{I+\gamma^5}{2}.
$$

The Clifford identities give $(\gamma^5)^2=I$ and
$\gamma^5\gamma^\mu=-\gamma^\mu\gamma^5$, hence
$P_L^2=P_L$, $P_R^2=P_R$, $P_LP_R=0$, and $P_L+P_R=I$.
They act in the existing recorded CAR representation.
:::

(sec-scalar-fields)=
## 4. Scalar Readouts and the Spatial Continuum Limit

:::{div} feynman-prose
A graph difference has two ingredients: the difference in field values and
how frequently the neighboring points occur. If twice as many walkers visit
one part of space, an unnormalized sum gives that region twice as much
weight. The limiting operator must remember this sampling density.

There is also a distinction between a spatial energy and a spacetime wave
operator. Squared spatial differences make a nonnegative Euclidean energy.
The Lorentzian construction needs signed directional moments and the causal
geometry of the previous chapter. We will keep those two calculations separate.
:::

### Geometry, moments, and the actual normalization

:::{prf:assumption} Local spatial sampling model
:label: assm-lqft-spatial-sampling

Supply a smooth $d$-dimensional Riemannian manifold $(X,g_R)$ and a fixed
query point $x$ with an injective normal-coordinate ball of radius $r>0$.
The claims uniform over a compact set require uniform versions of these
local bounds; compactness of the whole state space is not assumed. Let
$\rho$ be a positive probability density with respect to $d\mathrm{Vol}_{g_R}$.
Assume $\rho$, $\phi$, and the normal-coordinate volume Jacobian have four
bounded derivatives on the ball. For density correction also assume
$\rho\ge\rho_->0$ there.

Let $\kappa(u)=k(|u|^2)\ge0$ be a smooth radial kernel supported in
$|u|\le1$, with

$$
m_0=\int_{\mathbb R^d}\kappa(u)\,du>0,
\qquad
\int u_a u_b\kappa(u)\,du=m_2\delta_{ab},\qquad m_2>0.
$$

The parameter $\epsilon<r$ is a **length**. The associated heat bandwidth
would be $t=\epsilon^2$. Observations $X_1,\ldots,X_N$ have common marginal
$\rho\,d\mathrm{Vol}_{g_R}$. Independence, an aggregate covariance bound,
or a joint Poincare inequality will be specified in each variance claim.
For observations taken from the Fractal Gas, the marginal and the joint law
must be the ones to which the cited convergence result
applies. A quasi-stationary law, its conditioned history, and an invariant
law of another process cannot be interchanged.

The operator below includes every observation in the kernel ball. Applying
its limit to a recorded IG graph additionally requires that omitted edges,
recorded distances, and its actual weights produce an error tending to zero
after this operator's normalization. These comparisons are properties to
verify for that reconstruction. They are not changes to the particle algorithm.
:::

The analytic regularity already established in
{prf:ref}`thm-main-complete-cinf-geometric-gas-full` supplies fixed-parameter
expected-fitness derivatives under its stated density and normalization
hypotheses. Their use for $g_R$ is made explicit in
{prf:ref}`lem-cst-existing-spatial-regularity`. Distance comparisons can use
{prf:ref}`lem-cst-graph-distance-comparison` when its path approximation
hypothesis has been checked. Neither result asserts that arbitrary recorded
edge lengths equal the geodesic distances in the supplied manifold.

:::{prf:theorem} Unnormalized kernel Laplacian: bias and sampling error
:label: thm-laplacian-convergence

Under {prf:ref}`assm-lqft-spatial-sampling`, set

$$
(L_{N,\epsilon}\phi)(x)
 =\frac1{N\epsilon^{d+2}}\sum_{j=1}^N
 k\!\left(\frac{d_{g_R}(x,X_j)^2}{\epsilon^2}\right)
 [\phi(X_j)-\phi(x)].
$$

The deterministic mean has the limit

$$
L_\rho\phi
 =\frac{m_2}{2}\left(\rho\,\Delta_{g_R}\phi
                  +2\langle\nabla\rho,\nabla\phi\rangle_{g_R}\right)
 =\frac{m_2}{2\rho}\operatorname{div}_{g_R}
                 (\rho^2\nabla\phi),
\qquad
|\mathbb E L_{N,\epsilon}\phi(x)-L_\rho\phi(x)|
 \le C_b\epsilon^2.
$$

Here $\Delta=\operatorname{div}\nabla$. In particular this operator is not
$\rho^{-1}\operatorname{div}(\rho\nabla\phi)$. For constant density $\rho_0$
it is $(m_2\rho_0/2)\Delta\phi$.

Write $H_{\epsilon,x}(y)=\epsilon^{-d-2}
 k(d_{g_R}(x,y)^2/\epsilon^2)[\phi(y)-\phi(x)]$.
If the observations are independent, or satisfy

$$
\sum_{i,j=1}^N\operatorname{Cov}
 [H_{\epsilon,x}(X_i),H_{\epsilon,x}(X_j)]
 \le C_{\mathrm{cov}}N\epsilon^{-d-2},
$$

then $\operatorname{Var}(L_{N,\epsilon}\phi(x))
\le C_{\mathrm{cov}}/(N\epsilon^{d+2})$.
Alternatively, if their actual joint law satisfies a Poincare inequality
with an $N$-uniform constant $C_*$ for the full particle gradient, then

$$
\operatorname{Var}(L_{N,\epsilon}\phi(x))
 \le\frac{C_* C_H}{N\epsilon^{d+4}}.
$$

Consequently the latter route gives pointwise convergence in probability
when $\epsilon\to0$ and $N\epsilon^{d+4}\to\infty$. For example
$\epsilon=N^{-1/(d+8)}$ balances the bounds on squared bias and variance,
giving mean-square error $O(N^{-4/(d+8)})$ when their constants are uniform.
In heat-bandwidth notation the variance condition is
$Nt^{d/2+2}\to\infty$.
:::

:::{prf:proof}
Write $y=\exp_x(\epsilon u)$ and
$d\mathrm{Vol}_{g_R}(y)=\epsilon^d j_x(\epsilon u)du$. Set

$$
F_x(\xi)=[\phi(\exp_x\xi)-\phi(x)]\rho(\exp_x\xi),
\qquad G_x(\xi)=F_x(\xi)j_x(\xi).
$$

Then $\mathbb E L_{N,\epsilon}\phi(x)
=\epsilon^{-2}\int\kappa(u)G_x(\epsilon u)du$.
Normal coordinates give $F_x(0)=0$, $j_x(0)=1$, and $Dj_x(0)=0$.
The constant term vanishes; radial symmetry cancels the linear and cubic
terms. The quadratic term is

$$
\frac{m_2}{2}\operatorname{tr}D^2G_x(0)
 =\frac{m_2}{2}
   [\rho\Delta\phi+2\langle\nabla\rho,\nabla\phi\rangle](x).
$$

To make the remainder explicit, let $B_a$ bound the operator norm of
$D^aF_x$ and let $J_b$ bound that of $D^b j_x$ on the ball, for
$0\le a,b\le4$. Taylor's integral remainder and the product rule give

$$
C_b=\frac1{24}
 \left(\sum_{a=0}^4\binom4a B_aJ_{4-a}\right)
 \int\kappa(u)|u|^4du.
$$

These bounds are finite under the stated local hypotheses. For example the
$B_a$ follow from the product rule applied to the supplied derivative
bounds on $\phi\circ\exp_x-\phi(x)$ and $\rho\circ\exp_x$.
The divergence expression follows by differentiating $\rho^2\nabla\phi$.

Let $L_\phi$ bound $|\nabla\phi|$ on the ball. Since
$|\phi(y)-\phi(x)|\le L_\phi\epsilon$ on the support, a density bound
$\rho_+$ and $\mathrm{Vol}(B(x,\epsilon))\le V_0\epsilon^d$ give

$$
\mathbb E H_{\epsilon,x}(X_1)^2
 \le \rho_+V_0\|\kappa\|_\infty^2 L_\phi^2\epsilon^{-d-2}.
$$

Independence therefore gives the first variance estimate. For dependent
observations, expansion of the variance of $N^{-1}\sum_iH(X_i)$ gives
exactly the stated covariance condition.

For the Poincare route apply
$\operatorname{Var}(F)\le C_*\int\sum_i|\nabla_iF|^2d\pi_N$
to $F=N^{-1}\sum_iH_{\epsilon,x}(X_i)$; velocity derivatives are zero.
The resulting bound is $(C_*/N)\int|\nabla H_{\epsilon,x}|^2\rho\,d\mathrm{Vol}$.
If $G_0$ bounds the coordinate-to-Riemannian gradient comparison, then

$$
|\nabla H_{\epsilon,x}|
 \le G_0 L_\phi
   (\|\nabla\kappa\|_\infty+\|\kappa\|_\infty)
   \epsilon^{-d-2}
$$

on its support, after increasing $G_0$ to cover both terms. Hence one may
use

$$
C_H=\rho_+V_0G_0^2L_\phi^2
 (\|\nabla\kappa\|_\infty+\|\kappa\|_\infty)^2.
$$

The mean-square error is variance plus squared bias. The selected scaling
balances $\epsilon^4$ with $(N\epsilon^{d+4})^{-1}$ and proves the last
claims. $\square$
:::

:::{prf:remark} Using the established LSI and propagation estimates
:label: rem-lqft-existing-sampling-estimates

The full-gradient, $N$-uniform LSI in {prf:ref}`cor-n-uniform-lsi`, under
its structural hypotheses for the identified joint law $\pi_N$, provides
the Poincare input. Its sufficient law structures are a product of the
specified one-particle Gibbs laws, a whole-joint bounded log-density tilt
whose oscillation is uniform in $N$, or a joint potential with an
$N$-uniform positive Hessian lower bound. A nonconvex bounded perturbation
of a uniformly convex one-particle potential is treated before tensorizing;
the oscillation of a summed perturbation is not uniformly bounded merely
because each summand is. None of these joint-law identities follows from
exchangeability or propagation of chaos alone. In the convention

$$
\operatorname{Ent}_{\pi_N}(f^2)
 \le2C_*\int\sum_i
       (|\nabla_{x_i}f|^2+|\nabla_{v_i}f|^2)d\pi_N,
$$

expansion at $f=1+\delta F$ gives the Poincare constant $C_*$.
This recovers a usable spatial sampling estimate without assuming
independence. A velocity-only gradient estimate does not provide this bound
for spatial kernels. Coordinate and Riemannian gradients are compared on
the kernel support, with their uniform comparison factor included in the
constants above. If the joint law carries discrete status variables, the
additional status entropy in {prf:ref}`prop-kl-status-entropy` must also be
controlled, or the estimate must be restricted to the identified continuous
law. The same argument is recorded for spacetime kernels
in {prf:ref}`lem-cst-poincare-variance`.

If the actual one-particle marginal is $\rho_N$ rather than $\rho$, the
additional bias is $|\int H_{\epsilon,x}(\rho_N-\rho)d\mathrm{Vol}|$.
For any established Wasserstein-1 bound $W_1(\rho_N,\rho)\le\delta_N$,
the Kantorovich--Rubinstein estimate gives the explicit sufficient bound
$C\delta_N\epsilon^{-d-2}$. This is the test-function step used in
{doc}`../convergence_program/13_quantitative_error_bounds`; its applicable
law, time range, and rate must be retained. For example, a valid
$\delta_N=O(N^{-1/2})$ bound combined with the Poincare estimate allows
$\epsilon=N^{-\alpha}$ for $0<\alpha<1/(2d+4)$. No
$N^{-1}$ empirical squared-Wasserstein rate independent of dimension is
needed here.

All pointwise statements use a fixed query or an independent query sample.
Evaluation at one of the same dependent observations requires a conditional
estimate, uniform control over queries, or a direct joint argument.
Exchangeability alone supplies none of these. A same-sample action similarly
requires the joint quadrature conditions in
{prf:ref}`thm-cst-fractal-dalembertian-consistency`.
:::

:::{prf:proposition} Spatial energy and the limits of pointwise consistency
:label: prop-lqft-spatial-energy

For symmetric kernel weights and values $\phi_i=\phi(X_i)$, the finite
quadratic form associated with the unnormalized operator is

$$
\mathcal E_{N,\epsilon}(\phi)
 =-\frac1N\sum_i\phi_i(L_{N,\epsilon}\phi)_i
 =\frac1{2N^2\epsilon^{d+2}}
   \sum_{i,j}\kappa\!\left(\frac{\exp_{X_i}^{-1}X_j}{\epsilon}\right)
                    (\phi_j-\phi_i)^2.
$$

Its corresponding population form, integrated against
$\rho(x)\rho(y)d\mathrm{Vol}(x)d\mathrm{Vol}(y)$, converges to

$$
\mathcal E_\rho(\phi)
  =\frac{m_2}{2}\int_X\rho^2|\nabla\phi|_{g_R}^2d\mathrm{Vol}_{g_R}.
$$

Here $\phi$ has compact support, or the local Taylor bounds have an
integrable dominating envelope; normal-coordinate bounds hold uniformly
where the integrand is nonzero. The empirical energy has the same limit
whenever its shrinking-bandwidth pair quadrature error tends to zero.
Pointwise operator consistency alone does not assert this quadrature
condition, Mosco convergence of forms, spectral convergence, or convergence
of shortest-path distances.
:::

:::{prf:proof}
Pair the terms $(i,j)$ and $(j,i)$ to obtain the finite identity. In the
population double integral use $y=\exp_x(\epsilon u)$. Then
$\epsilon^{-1}(\phi(y)-\phi(x))\to d\phi_x(u)$,
$\rho(y)\to\rho(x)$, and $j_x(\epsilon u)\to1$.
Dominated convergence and the second-moment identity give
$(m_2/2)\int\rho^2|\nabla\phi|^2$. The empirical conclusion follows by
adding its explicitly assumed quadrature error. The stronger convergence
claims concern sequences of varying functions, spectra, or paths, none of
which are controlled by a calculation for one fixed test function. $\square$
:::

:::{prf:lemma} Sufficient same-sample energy estimates
:label: lem-lqft-energy-sampling

Assume the uniform local bounds in
{prf:ref}`prop-lqft-spatial-energy`, with compactly supported $\phi$ and
bounded sampling density on its kernel neighborhood. Write

$$
a_\epsilon(x,y)=\frac1{2\epsilon^{d+2}}
 k(d_{g_R}(x,y)^2/\epsilon^2)[\phi(y)-\phi(x)]^2.
$$

For independent samples from $\rho$, the energy satisfies

$$
\operatorname{Var}(\mathcal E_{N,\epsilon})
 \le C\left(\frac1N+\frac1{N^2\epsilon^d}\right),
\qquad
\mathbb E\mathcal E_{N,\epsilon}
 =(1-N^{-1})\iint a_\epsilon\rho(x)\rho(y)
                      d\mathrm{Vol}(x)d\mathrm{Vol}(y).
$$

For an exchangeable interacting joint law satisfying the full-gradient
Poincare bound with constant $C_*$, one instead has the sufficient estimate

$$
\operatorname{Var}(\mathcal E_{N,\epsilon})
 \le\frac{C C_*}{N\epsilon^{2d+2}}.
$$

If an applicable two-particle convergence estimate gives
$W_1(\pi_N^{(2)},\rho\otimes\rho)\le\delta_N^{(2)}$, the bias relative
to the population energy is bounded by
$C\delta_N^{(2)}\epsilon^{-d-1}+C/N$.
Thus $N\epsilon^{2d+2}\to\infty$ and
$\delta_N^{(2)}\epsilon^{-d-1}\to0$ suffice for the interacting
same-sample energy limit. With an established
$\delta_N^{(2)}=O(N^{-1/2})$, any
$\epsilon=N^{-\alpha}$ with $0<\alpha<1/(2d+2)$ meets these requirements.
The Poincare and two-particle bounds must hold for the same observation law.
:::

:::{prf:proof}
The local Lipschitz bound on $\phi$ implies
$|a_\epsilon|\le C\epsilon^{-d}$ and
$|\nabla_xa_\epsilon|+|\nabla_ya_\epsilon|
\le C\epsilon^{-d-1}$. The kernel support has volume $O(\epsilon^d)$.
Consequently $\sup_x\int a_\epsilon(x,y)\rho(y)d\mathrm{Vol}(y)\le C$
and $\iint a_\epsilon^2\rho(x)\rho(y)d\mathrm{Vol}(x)d\mathrm{Vol}(y)
\le C\epsilon^{-d}$.

In the variance expansion of $N^{-2}\sum_{i,j}a_\epsilon(X_i,X_j)$,
disjoint pairs are independent. There are $O(N^3)$ pairs sharing one index;
their covariance is bounded by the squared conditional-mean bound. There
are $O(N^2)$ pairs sharing both indices; their covariance is bounded by
the second moment. Division by $N^4$ proves the independent estimate.
The diagonal vanishes, giving the displayed expectation.

For the interacting estimate each particle gradient of the empirical energy
is bounded by $C/(N\epsilon^{d+1})$. Summing its square over the $N$
particles and applying Poincare proves the variance bound. Exchangeability
expresses the expectation through the common off-diagonal two-particle law.
The Lipschitz bound on $a_\epsilon$ and the Kantorovich--Rubinstein
inequality give the marginal bias, including the $N^{-1}$ diagonal
correction. The population energy is uniformly bounded by the first
integral estimate. The stated limits now follow by Chebyshev's inequality.
$\square$
:::

The classical [Belkin--Niyogi kernel analysis](https://misha.belkin-wang.org/papers/TT_JCSS_08.pdf)
provides pointwise consistency results for the stated manifold-sampling
models, including uniform independent sampling with a Gaussian heat kernel.
Its bandwidth, sign, density, and kernel conventions must be translated
before applying its formulas. The compactly supported-kernel calculation
above gives the normalization needed here directly; a pointwise theorem
from that paper does not supply the additional form or metric convergence
claims for the recorded graph.

### What density correction must normalize

:::{div} feynman-prose
Dividing by an estimate of the density removes the excess contribution of
crowded regions. But a second normalization fixes the total weight of a row,
and a factor of $\epsilon^{-2}$ fixes the units of a second derivative.
Leaving out either step changes the operator. We can see every factor by
first doing the calculation with the exact density.
:::

:::{prf:definition} Density correction and row normalization
:label: def-density-corrected-laplacian

For the kernel graph define

$$
w_{ij}=\frac1{N\epsilon^{d+2}}
       k\!\left(\frac{d_{g_R}(X_i,X_j)^2}{\epsilon^2}\right),
\quad q_i=\sum_jw_{ij},\quad
\widetilde w_{ij}=\frac{w_{ij}}{q_iq_j},\quad
\widetilde q_i=\sum_j\widetilde w_{ij}.
$$

Rows with zero degree are excluded from a pointwise reconstruction.
The unnormalized corrected operator is
$\widetilde L_{N,\epsilon}\phi_i
=\sum_j\widetilde w_{ij}(\phi_j-\phi_i)$.
The completed row-normalized reconstruction is

$$
L^{\mathrm{corr}}_{N,\epsilon}\phi_i
 =\frac{2m_0}{m_2\epsilon^2}
   \frac{\sum_j\widetilde w_{ij}(\phi_j-\phi_i)}
        {\sum_j\widetilde w_{ij}}.
$$

This is a specified post-processing operator on a sampled graph. It is not
an assertion that the recorded algorithm already uses these weights.
When used with an estimated positive density $\widehat\rho_i$, the same
row ratio can instead use weights
$k(d_{g_R}(X_i,X_j)^2/\epsilon^2)/(\widehat\rho_i\widehat\rho_j)$.
In particular degrees give $\widehat\rho_i=\epsilon^2q_i/m_0$.
:::

:::{prf:proposition} Density-corrected limit with sufficient sampling control
:label: prop-density-corrected-limit

Under {prf:ref}`assm-lqft-spatial-sampling`, first use the exact density
$\widehat\rho=\rho$ in the row-normalized reconstruction and a fixed query
$x$. Under the independence/covariance or full-gradient Poincare sampling
conditions that make the following numerator and denominator converge,

$$
L^{\mathrm{corr}}_{N,\epsilon}\phi(x)
 \xrightarrow{\mathbb P}\Delta_{g_R}\phi(x).
$$

For the Poincare route $N\epsilon^{d+4}\to\infty$ is sufficient, with
bounds on $\rho^{-1}$ and its derivatives on the kernel ball. Replacing
$\rho$ by a positive estimator preserves this limit if its relative error
on that ball is $o_{\mathbb P}(\epsilon)$ and its denominator remains
positive. These requirements include the joint dependence of a degree
estimate and the points at which it is used; a pointwise degree estimate
alone does not establish them.

The operator $\widetilde L_{N,\epsilon}$ without row normalization has
a different scale. With ideal degrees $q(y)=m_0\rho(y)/\epsilon^2$ and the
same sufficient sampling bounds,

$$
\epsilon^{-4}\widetilde L_{N,\epsilon}\phi(x)
 \xrightarrow{\mathbb P}
 \frac{m_2}{2m_0^2\rho(x)}\Delta_{g_R}\phi(x).
$$

Thus it tends to zero before rescaling, and the rescaled limit still has a
spatially varying density factor. A scalar bandwidth factor alone does not
turn this particular unnormalized correction into $\Delta_{g_R}$ for
nonconstant $\rho$.
:::

:::{prf:proof}
For exact density the common query factor $\rho(x)^{-1}$ cancels from the
row ratio. Write

$$
A_{N,\epsilon}(x)=\frac1{N\epsilon^{d+2}}\sum_j
 \frac{k(d_{g_R}(x,X_j)^2/\epsilon^2)}{\rho(X_j)}
 [\phi(X_j)-\phi(x)],
\qquad
B_{N,\epsilon}(x)=\frac1{N\epsilon^d}\sum_j
 \frac{k(d_{g_R}(x,X_j)^2/\epsilon^2)}{\rho(X_j)}.
$$

The row-normalized operator is $(2m_0/m_2)A_{N,\epsilon}/B_{N,\epsilon}$.
The density cancels from both population integrals. The normal-coordinate
calculation gives

$$
\mathbb EA_{N,\epsilon}=\frac{m_2}{2}\Delta\phi(x)+O(\epsilon^2),
\qquad
\mathbb EB_{N,\epsilon}=m_0+O(\epsilon^2).
$$

The same gradient proof as in {prf:ref}`thm-laplacian-convergence`, with
bounded $\rho^{-1}$ and $\nabla\rho^{-1}$, gives variances bounded by
$C/(N\epsilon^{d+4})$ and $C/(N\epsilon^{d+2})$, respectively.
Hence $A$ and $B$ converge in probability, $B$ stays bounded away from
zero with probability tending to one, and their ratio has the stated limit.
Independence gives the corresponding, weaker bandwidth requirements by
bounding the second moments instead of gradients.

For an estimated density let
$\delta=\sup_{B(x,\epsilon)}|\widehat\rho/\rho-1|<1/2$.
Its reciprocal weights differ multiplicatively from the exact ones by
at most $2\delta$. Normalizing two such nonnegative rows changes their
probability weights in total variation by at most $8\delta$. Since
$|\phi(X_j)-\phi(x)|\le L_\phi\epsilon$, the resulting operator error is
at most $16m_0L_\phi\delta/(m_2\epsilon)$, increasing the constant if
necessary for the total-variation convention. It tends to zero under the
stated condition. No independence of the density estimator is needed for
this deterministic comparison.

Finally ideal degrees give the exact algebra

$$
\widetilde L_{N,\epsilon}\phi(x)
 =\frac{\epsilon^4}{m_0^2\rho(x)}A_{N,\epsilon}(x).
$$

The last assertion follows from the already proved limit for $A$.
For empirical degrees this rescaled assertion likewise needs a sufficiently
small reconstruction error; degree consistency without its rate does not
justify multiplication by a diverging normalization. $\square$
:::

(sec-complete-framework)=
## 5. Recorded Operators and Their Evolution

:::{div} feynman-prose
The direct formulation now has a precise path from records to measurements.
Reconstruct the inputs, form the full invariant coordinates, and push the
recorded law through that map. The orbit and observable-space isomorphisms
preserve the direct correlations exactly. Channel averages are then computed
as specified functions of these coordinates and the recorded auxiliary data.

The complete transition record also determines the conditional expectations
used in {prf:ref}`thm-lqft-instantiated-word-evolution`. Alternating
insertions faithfully realize the exterior algebra on replica observables;
their adjoint contractions complete the CAR. Propagating the recorded modes
gives both the stated covariance determinants and the completely positive
CAR evolution of {prf:ref}`thm-lqft-record-car-channel`.

Regional covariance matrices now determine the CAR locality coefficients,
and the complete update determines their propagation. The existing LSI bound
controls the norms of the recorded modes in its specified Sobolev domain.
:::

The recorded internal-symmetry observables are developed in
{doc}`04_standard_model`; the dynamical Yang--Mills questions are treated
in {doc}`05_yang_mills_noether`.
Their continuum applications must retain the conditions stated here.
