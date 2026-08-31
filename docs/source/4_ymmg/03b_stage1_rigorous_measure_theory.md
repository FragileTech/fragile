# Stage I: Rigorous finite-volume measure theory for the Fractal-Gas reformulation of `03b`

## Purpose

This document gives the **Stage I** replacement package for the revised `03b` paper. Its purpose is to upgrade the current finite random gauge construction from

- a measurable pushforward law on realized gauge data,

into

- a **fibered finite-volume gauge measure theory** with
  1. a precise configuration-space theorem,
  2. disintegration of the induced gauge law over realized regulators,
  3. a gauge-invariance theorem for the conditional laws,
  4. an **exact effective-action representation** under an explicit absolute-continuity hypothesis, and
  5. a **Wilson-comparison theorem** under an explicit remainder/oscillation hypothesis.

This is the correct measure-theoretic foundation for building a nonabelian Euclidean gauge theory on the Fractal-Gas regulator.

The package below is divided into two parts.

- **Unconditional part:** everything that follows from measurability, compactness of the gauge fibers, and standard disintegration theory.
- **Conditional part:** theorems that require explicit hypotheses, stated cleanly and proved rigorously from those hypotheses.

That division is essential. It keeps the document fully honest:

- what is already true from finite-dimensional measure theory is proved without gaps,
- what still depends on model-specific estimates is isolated as a hypothesis, so the downstream papers know exactly what remains to be established.

---

## 1. Role of Stage I inside the new `03b`

The revised `03b` already moved away from the old deterministic flat regulator and introduced:

- a **realized Fractal regulator** extracted from each swarm history,
- realized edge transports,
- plaquette holonomies and a Wilson action,
- and a pushforward finite gauge law.

That is the right first move, but it is not yet enough to build Yang--Mills QFT on top of it. The missing piece is a finite-volume measure theory with the same level of precision one uses for ordinary lattice gauge theory.

The precise gap is this:

1. the induced gauge law is a pushforward measure, but not yet written as a gauge measure relative to a standard reference measure,
2. no exact effective action has been defined,
3. no rigorous comparison theorem has been proved between the induced law and the Wilson law,
4. and no fiberwise gauge-invariance statement has been isolated at the level of the conditional laws.

Stage I fixes all four problems.

---

## 2. Fixed finite horizon and basic objects

Fix once and for all:

- a swarm size `N`,
- a recorded horizon `T`,
- a compact gauge group `G`, which for the target paper should be taken to be `SU(3)` (or `SU(2)` in a technically easier intermediate draft).

Let

\[
(\Omega_{N,T},\mathcal F_{N,T},\mathbb P_{N,T})
\]

be the swarm-history probability space furnished by the stochastic particle dynamics.

A history \(\omega\in\Omega_{N,T}\) determines a realized Fractal regulator:

- a finite vertex set \(V(\omega)\),
- a finite family \(E^+(\omega)\) of **stored positively oriented** edges,
- a finite family \(P(\omega)\) of oriented plaquettes,
- local geometric data such as edge types, incidence maps, plaquette areas, and any auxiliary couplings needed to evaluate the Wilson action.

We write the total realized regulator data as

\[
R(\omega)\in\mathscr R_{N,T},
\]

where \(\mathscr R_{N,T}\) is defined below.

For each stored positive edge \(e\in E^+(\omega)\), the revised `03b` already assigns a group element

\[
U_e(\omega)\in G.
\]

The transport on the reversed orientation is defined by

\[
U_{\bar e}(\omega)=U_e(\omega)^{-1}.
\]

Thus the true free gauge variables live on the stored positive edges only.

---

## 3. Regulator-data space

### Definition 3.1 (realized regulator-data space)

Let \(M_V(N,T),M_E(N,T),M_P(N,T)\) be deterministic upper bounds on the numbers of vertices, stored positive edges, and plaquettes that can occur at size `N` and horizon `T`.

Define \(\mathscr R_{N,T}\) to be the space of all tuples

\[
R=(m_V,m_E,m_P,\iota_V,\iota_E,\iota_P,\mathbf a,\mathbf A,\mathbf t,\mathbf c)
\]

such that:

- \(0\le m_V\le M_V(N,T)\), \(0\le m_E\le M_E(N,T)\), \(0\le m_P\le M_P(N,T)\),
- \(\iota_V\) records the finite vertex labels,
- \(\iota_E\) records the incidence/orientation data of the stored positive edges,
- \(\iota_P\) records the plaquette boundary data in terms of edge indices,
- \(\mathbf a\in [0,\infty)^{m_E}\) records the edge lengths/couplings needed by the finite action,
- \(\mathbf A\in [0,\infty)^{m_P}\) records the plaquette areas,
- \(\mathbf t\) records the edge/plaquette types (CST, IG, IA, etc.),
- \(\mathbf c\) records any additional bounded local combinatorial/geometric data needed to define the finite Wilson action.

Equip \(\mathscr R_{N,T}\) with the σ-algebra inherited from the finite disjoint union of Euclidean spaces times finite label sets.

### Proposition 3.2 (standard Borelness of the regulator space)

The space \(\mathscr R_{N,T}\) is a standard Borel space.

#### Proof

For fixed integers \((m_V,m_E,m_P)\), the data listed above live in a finite product of:

- finite sets,
- Euclidean spaces,
- and compact intervals.

Hence each such piece is a standard Borel space. Since the ranges of \(m_V,m_E,m_P\) are finite, \(\mathscr R_{N,T}\) is a finite disjoint union of standard Borel spaces, and therefore itself standard Borel. ∎

---

## 4. Gauge fibers over a realized regulator

### Definition 4.1 (fiber gauge configuration space)

Let \(R\in\mathscr R_{N,T}\) have stored positive edge set \(E^+(R)\) with cardinality \(m_E(R)\). The corresponding finite gauge fiber is

\[
\mathscr U(R):=G^{E^+(R)}\cong G^{m_E(R)}.
\]

An element \(U\in\mathscr U(R)\) is a family of edge transports \(\{U_e\}_{e\in E^+(R)}\). Reversed-edge transports are understood as inverses and are not stored independently.

### Proposition 4.2 (compactness and Polish structure of the fibers)

For every \(R\in\mathscr R_{N,T}\), the fiber \(\mathscr U(R)\) is a compact metrizable group and hence a compact Polish space.

#### Proof

The compact gauge group \(G\) is compact metrizable. A finite Cartesian product of compact metrizable spaces is compact metrizable. Since \(\mathscr U(R)\cong G^{m_E(R)}\) with finite \(m_E(R)\), the conclusion follows. ∎

### Definition 4.3 (total fibered gauge space)

Define the total finite-volume gauge space

\[
\mathscr X_{N,T}:=\{(R,U):R\in \mathscr R_{N,T},\ U\in \mathscr U(R)\}.
\]

Equip \(\mathscr X_{N,T}\) with the σ-algebra generated by:

- the projection \((R,U)\mapsto R\), and
- all fiberwise Borel sets.

Equivalently, one may embed each fiber into a fixed ambient space \(G^{M_E(N,T)}\) by padding unused coordinates with the identity and view \(\mathscr X_{N,T}\) as a measurable subset of

\[
\mathscr R_{N,T}\times G^{M_E(N,T)}.
\]

### Proposition 4.4 (standard Borelness of the total gauge space)

The total gauge space \(\mathscr X_{N,T}\) is a standard Borel space.

#### Proof

Choose the identity-padding embedding described above:

\[
\iota_R:\mathscr U(R)\hookrightarrow G^{M_E(N,T)}.
\]

Then

\[
\mathscr X_{N,T}
\cong
\{(R,U)\in \mathscr R_{N,T}\times G^{M_E(N,T)}:U\in \iota_R(\mathscr U(R))\}.
\]

The ambient product \(\mathscr R_{N,T}\times G^{M_E(N,T)}\) is standard Borel because both factors are standard Borel. The subset above is measurable by construction, hence standard Borel with the trace σ-algebra. ∎

---

## 5. Wilson action on the fibers

### Definition 5.1 (fiberwise Wilson action)

Let \(R\in\mathscr R_{N,T}\) and \(U\in\mathscr U(R)\). For each plaquette \(P\in P(R)\), let

\[
U_P(U):=\prod_{e\in\partial P} U_e^{\epsilon(P,e)}
\]

be the ordered plaquette holonomy, where \(\epsilon(P,e)=\pm 1\) records whether the stored orientation of \(e\) agrees with the boundary orientation of \(P\).

Define the fiberwise Wilson action by

\[
S_W(R,U):=\beta(R)\sum_{P\in P(R)}\Bigl(1-\frac{1}{N_c}\Re\operatorname{Tr}(U_P(U))\Bigr),
\]

where:

- \(N_c\) is the dimension of the defining representation of \(G\),
- \(\beta(R)\ge 0\) is any measurable inverse-coupling parameter attached to the realized regulator \(R\).

### Proposition 5.2 (measurability and boundedness of local Wilson observables)

For each fixed finite loop \(\gamma\) or plaquette \(P\), the maps

\[
(R,U)\mapsto W_\gamma(R,U),
\qquad
(R,U)\mapsto \Re\operatorname{Tr}(U_P(U)),
\qquad
(R,U)\mapsto S_W(R,U)
\]

are measurable on \(\mathscr X_{N,T}\). Moreover, all Wilson loops are bounded by

\[
|W_\gamma(R,U)|\le N_c.
\]

#### Proof

Each Wilson loop is a finite product of coordinate projections \(U_e\), matrix inversion, matrix multiplication, and trace. All of these are continuous operations on the compact Lie group \(G\) and its matrix model, hence Borel. Therefore each Wilson loop and each plaquette trace is measurable.

The Wilson action is a finite sum of measurable plaquette terms with measurable coefficient \(\beta(R)\), so it is measurable.

The operator norm of each \(U_e\in G\subset U(N_c)\) is 1, so the same holds for every finite product \(U_\gamma\). Hence

\[
|\operatorname{Tr}(U_\gamma)|\le N_c.
\]

This gives the claimed bound. ∎

---

## 6. Induced finite gauge law and disintegration

### Definition 6.1 (regulator extraction and gauge construction)

Let

\[
\mathfrak R^{(N,T)}:\Omega_{N,T}\to \mathscr R_{N,T}
\]

be the measurable map extracting the realized Fractal regulator data from a swarm history.

Let

\[
\mathfrak U^{(N,T)}:\Omega_{N,T}\to \bigsqcup_{R\in\mathscr R_{N,T}}\mathscr U(R)
\]

be the measurable map assigning the realized edge transports.

Define the combined gauge-construction map

\[
\Xi^{(N,T)}(\omega):=\bigl(\mathfrak R^{(N,T)}(\omega),\mathfrak U^{(N,T)}(\omega)\bigr)\in \mathscr X_{N,T}.
\]

### Theorem 6.2 (induced finite-volume gauge law)

The pushforward measure

\[
\mu_{N,T}:=\Xi^{(N,T)}_{\#}\mathbb P_{N,T}\in \mathcal P(\mathscr X_{N,T})
\]

is well defined. For every bounded measurable observable \(\Phi:\mathscr X_{N,T}\to\mathbb R\),

\[
\int_{\mathscr X_{N,T}}\Phi(R,U)\,\mu_{N,T}(dR,dU)
=
\mathbb E_{\mathbb P_{N,T}}\bigl[\Phi(\Xi^{(N,T)}(\omega))\bigr].
\]

#### Proof

The map \(\Xi^{(N,T)}\) is measurable by assumption, and \(\mathscr X_{N,T}\) is standard Borel by Proposition 4.4. Therefore the pushforward measure is well defined. The integral identity is the defining property of pushforward measures. ∎

### Corollary 6.3 (existence of all finite Wilson moments)

Every bounded gauge observable generated by finitely many Wilson loops and plaquette traces is \(\mu_{N,T}\)-integrable. In particular, all finite joint moments of Wilson loops exist.

#### Proof

By Proposition 5.2, Wilson loops are bounded measurable functions on \(\mathscr X_{N,T}\), hence integrable under any probability measure on \(\mathscr X_{N,T}\), in particular under \(\mu_{N,T}\). Finite products of bounded measurable functions are again bounded and measurable. ∎

### Theorem 6.4 (disintegration over realized regulators)

Let

\[
\lambda_{N,T}:=\mathfrak R^{(N,T)}_{\#}\mathbb P_{N,T}\in \mathcal P(\mathscr R_{N,T}).
\]

Then there exists a \(\lambda_{N,T}\)-a.e. uniquely determined measurable family of probability measures

\[
R\mapsto \mu_{N,T}^R\in\mathcal P(\mathscr U(R))
\]

such that for every bounded measurable \(\Phi\) on \(\mathscr X_{N,T}\),

\[
\int_{\mathscr X_{N,T}}\Phi(R,U)\,\mu_{N,T}(dR,dU)
=
\int_{\mathscr R_{N,T}}
\left(
\int_{\mathscr U(R)}\Phi(R,U)\,\mu_{N,T}^R(dU)
\right)
\lambda_{N,T}(dR).
\]

#### Proof

The projection

\[
\pi:\mathscr X_{N,T}\to \mathscr R_{N,T},\qquad \pi(R,U)=R
\]

is measurable between standard Borel spaces. Hence the Rokhlin disintegration theorem applies to the probability measure \(\mu_{N,T}\) relative to \(\pi\), producing a measurable probability kernel \(R\mapsto \mu_{N,T}^R\) supported on the fiber \(\mathscr U(R)\). The stated integral identity is exactly the disintegration formula. Uniqueness holds up to \(\lambda_{N,T}\)-null sets. ∎

### Remark 6.5 (what has been proved unconditionally)

Up to this point, everything is unconditional. The only inputs are:

- finite horizon and finite swarm size,
- measurability of the regulator and gauge-construction maps,
- compactness of the gauge group.

No Gibbs structure, no Yang--Mills comparison, and no absolute continuity have been assumed yet.

---

## 7. Gauge group action on each fiber

### Definition 7.1 (fiber gauge group)

For each realized regulator \(R\), define the local gauge group

\[
\mathcal G(R):=G^{V(R)}.
\]

For \(g\in\mathcal G(R)\) and \(U\in\mathscr U(R)\), define

\[
(g\cdot U)_e := g_{t(e)}U_e g_{s(e)}^{-1},
\]

where \(s(e)\) and \(t(e)\) are the source and target vertices of the stored oriented edge \(e\).

### Proposition 7.2 (fiberwise gauge action is measurable)

For each fixed \(R\), the map

\[
\mathcal G(R)\times \mathscr U(R)\to \mathscr U(R),\qquad (g,U)\mapsto g\cdot U
\]

is continuous.

#### Proof

It is a finite product of continuous group operations in the compact Lie group \(G\): multiplication and inversion. ∎

### Definition 7.3 (gauge-invariant observable)

A measurable function \(\Phi\) on \(\mathscr X_{N,T}\) is called gauge-invariant if for every \(R\in\mathscr R_{N,T}\), every \(g\in\mathcal G(R)\), and every \(U\in\mathscr U(R)\),

\[
\Phi(R,g\cdot U)=\Phi(R,U).
\]

Wilson loops and the Wilson action are gauge-invariant in this sense.

---

## 8. Fiberwise gauge invariance of the induced law

The following theorem is rigorous, but it needs a model-side symmetry hypothesis.

### Assumption 8.1 (gauge-equivariant lift on the swarm history space)

For each realized regulator \(R\) and each local gauge map \(g\in\mathcal G(R)\), there exists a measurable transformation

\[
\widetilde g:\Omega_{N,T}\to\Omega_{N,T}
\]

such that:

1. **regulator invariance**
   \[
   \mathfrak R^{(N,T)}(\widetilde g\omega)=\mathfrak R^{(N,T)}(\omega),
   \]
2. **gauge-equivariance of the edge assignment**
   \[
   \mathfrak U^{(N,T)}(\widetilde g\omega)=g\cdot \mathfrak U^{(N,T)}(\omega),
   \]
3. **invariance of the swarm law**
   \[
   \mathbb P_{N,T}\circ \widetilde g^{-1}=\mathbb P_{N,T}.
   \]

This is the exact finite-volume hypothesis expressing that the underlying history law has the local frame redundancy that should descend to gauge invariance of the induced gauge law.

### Theorem 8.2 (fiberwise gauge invariance of the induced gauge law)

Under Assumption 8.1, for \(\lambda_{N,T}\)-almost every realized regulator \(R\), the conditional law \(\mu_{N,T}^R\) is gauge-invariant:

\[
\mu_{N,T}^R(A)=\mu_{N,T}^R(g\cdot A)
\]

for every Borel set \(A\subseteq \mathscr U(R)\) and every \(g\in\mathcal G(R)\).

Equivalently, for every bounded measurable \(f:\mathscr U(R)\to\mathbb R\),

\[
\int f(U)\,\mu_{N,T}^R(dU)
=
\int f(g\cdot U)\,\mu_{N,T}^R(dU).
\]

#### Proof

Fix a bounded measurable \(f\) and define the corresponding observable on \(\Omega_{N,T}\):

\[
F_f(\omega):=f\bigl(\mathfrak U^{(N,T)}(\omega)\bigr)\mathbf 1_{\{\mathfrak R^{(N,T)}(\omega)=R\}}.
\]

By Assumption 8.1,

\[
F_f(\widetilde g\omega)
=
 f\bigl(g\cdot \mathfrak U^{(N,T)}(\omega)\bigr)\mathbf 1_{\{\mathfrak R^{(N,T)}(\omega)=R\}}.
\]

Since \(\mathbb P_{N,T}\) is invariant under \(\widetilde g\),

\[
\mathbb E[F_f]
=
\mathbb E[F_f\circ \widetilde g].
\]

Using the disintegration identity from Theorem 6.4 and cancelling the common indicator of the regulator fiber, this yields

\[
\int_{\mathscr U(R)} f(U)\,\mu_{N,T}^R(dU)
=
\int_{\mathscr U(R)} f(g\cdot U)\,\mu_{N,T}^R(dU)
\]

for \(\lambda_{N,T}\)-a.e. \(R\). This is exactly gauge invariance of the conditional law. ∎

### Corollary 8.3 (gauge-invariant observables descend to the quotient)

Under Assumption 8.1, expectations of gauge-invariant observables depend only on the gauge orbit of the configuration in each fiber.

#### Proof

If \(\Phi(R,\cdot)\) is gauge-invariant, then \(\Phi(R,g\cdot U)=\Phi(R,U)\) pointwise. The claim follows immediately from Theorem 8.2. ∎

---

## 9. Fiberwise Haar reference measures

### Definition 9.1 (fiberwise reference Haar measure)

For each realized regulator \(R\), let

\[
\nu_R:=\bigotimes_{e\in E^+(R)} d\mathrm{Haar}(U_e)
\in \mathcal P(\mathscr U(R)).
\]

This is the product Haar measure on the fiber \(\mathscr U(R)=G^{E^+(R)}\).

### Proposition 9.2 (gauge invariance of the fiber Haar measures)

For every realized regulator \(R\) and every local gauge map \(g\in\mathcal G(R)\),

\[
(g_{\#}\nu_R)=\nu_R.
\]

#### Proof

For a single edge variable, left and right multiplication by fixed group elements preserve Haar measure. Since the fiber measure is a finite product Haar measure and the gauge action acts on each edge by left multiplication at the target and right multiplication by the inverse at the source, the whole product measure is invariant. ∎

### Proposition 9.3 (measurable reference-measure kernel)

The assignment

\[
R\mapsto \nu_R
\]

is a measurable probability kernel from \(\mathscr R_{N,T}\) to \(\mathscr X_{N,T}\).

#### Proof

Use the fixed ambient embedding into \(G^{M_E(N,T)}\): for each \(R\), the measure \(\nu_R\) is the product Haar measure on the first \(m_E(R)\) coordinates and the Dirac mass at the identity on the remaining padded coordinates. Since \(m_E(R)\) is measurable in \(R\), this defines a measurable probability kernel. ∎

---

## 10. Exact effective action under absolute continuity

The next theorem is rigorous once a single explicit hypothesis is assumed.

### Assumption 10.1 (fiberwise absolute continuity)

For \(\lambda_{N,T}\)-almost every regulator \(R\), the conditional induced gauge law \(\mu_{N,T}^R\) is absolutely continuous with respect to the fiber Haar measure \(\nu_R\):

\[
\mu_{N,T}^R\ll \nu_R.
\]

### Theorem 10.2 (exact effective-action representation)

Under Assumption 10.1, there exists a jointly measurable nonnegative density

\[
\rho_{N,T}(R,U)
=\frac{d\mu_{N,T}^R}{d\nu_R}(U)
\]

such that

\[
\mu_{N,T}(dR,dU)=\lambda_{N,T}(dR)\,\rho_{N,T}(R,U)\,\nu_R(dU).
\]

Define the exact effective action

\[
S_{\mathrm{eff}}^{(N,T)}(R,U):=
\begin{cases}
-\log \rho_{N,T}(R,U), & \rho_{N,T}(R,U)>0,\\[4pt]
+\infty, & \rho_{N,T}(R,U)=0.
\end{cases}
\]

Then

\[
\mu_{N,T}(dR,dU)=\lambda_{N,T}(dR)\,e^{-S_{\mathrm{eff}}^{(N,T)}(R,U)}\,\nu_R(dU).
\]

#### Proof

The disintegration theorem gives the measurable kernel \(R\mapsto\mu_{N,T}^R\). The family \(R\mapsto \nu_R\) is also a measurable kernel by Proposition 9.3. Assumption 10.1 says that fiberwise absolute continuity holds for \(\lambda_{N,T}\)-a.e. \(R\). Therefore the Radon--Nikodym theorem for measurable kernels yields a jointly measurable density \(\rho_{N,T}(R,U)\) satisfying

\[
\mu_{N,T}^R(dU)=\rho_{N,T}(R,U)\nu_R(dU)
\]

for \(\lambda_{N,T}\)-a.e. \(R\). Substituting this into the disintegration formula proves the claimed representation. The exponential form is simply the definition of \(S_{\mathrm{eff}}^{(N,T)}\). ∎

### Remark 10.3 (what has and has not been achieved)

Theorem 10.2 is the exact measure-theoretic upgrade from a pushforward law to a finite-volume gauge measure with an exact effective action. However, the theorem **does not identify**

\[
S_{\mathrm{eff}}^{(N,T)}
\]

with the Wilson action.

That identification, or even approximation, is a separate theorem.

---

## 11. Pure Wilson fiber measures

### Definition 11.1 (fiberwise Wilson measure)

For each realized regulator \(R\), define the pure Wilson partition function

\[
Z_W(R):=\int_{\mathscr U(R)} e^{-S_W(R,U)}\,\nu_R(dU).
\]

Since \(\mathscr U(R)\) is compact and \(S_W(R,\cdot)\) is bounded below and measurable, \(0<Z_W(R)<\infty\).

Define the fiberwise Wilson measure

\[
\mu_{W}^R(dU):=Z_W(R)^{-1}e^{-S_W(R,U)}\,\nu_R(dU).
\]

Define the mixed Wilson law on the total space by keeping the same regulator law:

\[
\mu_{W}^{(N,T)}(dR,dU):=\lambda_{N,T}(dR)\,\mu_W^R(dU).
\]

### Proposition 11.2 (gauge invariance of the Wilson fibers)

For every realized regulator \(R\), the Wilson measure \(\mu_W^R\) is gauge-invariant.

#### Proof

The Haar measure \(\nu_R\) is gauge-invariant by Proposition 9.2, and the Wilson action \(S_W(R,\cdot)\) is gauge-invariant by construction. Therefore \(e^{-S_W}\nu_R\) is gauge-invariant, and so is its normalization. ∎

---

## 12. Wilson comparison theorem

This is the key Stage I theorem. It tells you exactly what must be proved to claim that the induced swarm gauge law is a Wilson-type gauge law.

### Assumption 12.1 (Wilson comparison remainder)

Assume there exist measurable functions

\[
c_{N,T}:\mathscr R_{N,T}\to\mathbb R,
\qquad
\Delta_{N,T}:\mathscr X_{N,T}\to\mathbb R,
\]

such that for \(\lambda_{N,T}\)-a.e. \(R\) and \(\nu_R\)-a.e. \(U\),

\[
S_{\mathrm{eff}}^{(N,T)}(R,U)=S_W(R,U)+\Delta_{N,T}(R,U)+c_{N,T}(R).
\]

Assume also that \(\Delta_{N,T}(R,\cdot)\) is gauge-invariant for \(\lambda_{N,T}\)-a.e. \(R\), and that its oscillation is bounded by a measurable error profile \(\varepsilon_{N,T}(R)\ge 0\):

\[
\operatorname{osc}_U \Delta_{N,T}(R,\cdot)
:=
\sup_{U\in\mathscr U(R)}\Delta_{N,T}(R,U)-\inf_{U\in\mathscr U(R)}\Delta_{N,T}(R,U)
\le \varepsilon_{N,T}(R).
\]

### Theorem 12.2 (fiberwise comparison with the Wilson measure)

Under Assumptions 10.1 and 12.1, for \(\lambda_{N,T}\)-a.e. regulator \(R\), the induced fiber measure \(\mu_{N,T}^R\) and the Wilson fiber measure \(\mu_W^R\) are mutually absolutely continuous, with density ratio satisfying the uniform bound

\[
e^{-\varepsilon_{N,T}(R)}
\le
\frac{d\mu_{N,T}^R}{d\mu_W^R}(U)
\le
e^{\varepsilon_{N,T}(R)}
\qquad
\text{for }\mu_W^R\text{-a.e. }U.
\]

Consequently,

\[
\|\mu_{N,T}^R-\mu_W^R\|_{\mathrm{TV}}
\le \sinh\bigl(\varepsilon_{N,T}(R)\bigr)
\le e^{\varepsilon_{N,T}(R)}-1.
\]

For every bounded measurable observable \(f\) on \(\mathscr U(R)\),

\[
\left|\int f\,d\mu_{N,T}^R-\int f\,d\mu_W^R\right|
\le 2\|f\|_{\infty}\,\sinh\bigl(\varepsilon_{N,T}(R)\bigr).
\]

#### Proof

Fix \(R\) in the full-measure set where the assumptions hold. By Theorem 10.2 and Definition 11.1,

\[
\frac{d\mu_{N,T}^R}{d\mu_W^R}(U)
=
\frac{Z_W(R)}{e^{c_{N,T}(R)}}e^{-\Delta_{N,T}(R,U)}.
\]

To bound the normalization factor, let

\[
m(R):=\inf_U \Delta_{N,T}(R,U),
\qquad
M(R):=\sup_U \Delta_{N,T}(R,U),
\]

so \(M(R)-m(R)\le \varepsilon_{N,T}(R)\). Then

\[
e^{-M(R)}
\le
\int e^{-\Delta_{N,T}(R,U)}\,\mu_W^R(dU)
\le
e^{-m(R)}.
\]

Since

\[
\frac{Z_W(R)}{e^{c_{N,T}(R)}}
=
\left(\int e^{-\Delta_{N,T}(R,U)}\,\mu_W^R(dU)\right)^{-1},
\]

we obtain

\[
e^{m(R)}
\le
\frac{Z_W(R)}{e^{c_{N,T}(R)}}
\le
e^{M(R)}.
\]

Therefore

\[
e^{m(R)-M(R)}
\le
\frac{d\mu_{N,T}^R}{d\mu_W^R}(U)
\le
e^{M(R)-m(R)}.
\]

Using \(M(R)-m(R)\le \varepsilon_{N,T}(R)\), this yields the density-ratio bound.

For the total-variation estimate,

\[
\|\mu_{N,T}^R-\mu_W^R\|_{\mathrm{TV}}
=
\frac12\int\left|\frac{d\mu_{N,T}^R}{d\mu_W^R}-1\right|d\mu_W^R
\le \frac12\max\{e^{\varepsilon}-1,1-e^{-\varepsilon}\}
\le \sinh(\varepsilon),
\]

where \(\varepsilon=\varepsilon_{N,T}(R)\). The observable bound follows from the standard inequality

\[
\left|\int f\,d\mu-\int f\,d\nu\right|\le 2\|f\|_\infty\|\mu-\nu\|_{\mathrm{TV}}.
\]

∎

### Corollary 12.3 (mixture-level Wilson comparison)

Under Assumptions 10.1 and 12.1,

\[
\|\mu_{N,T}-\mu_W^{(N,T)}\|_{\mathrm{TV}}
\le
\int_{\mathscr R_{N,T}} \sinh\bigl(\varepsilon_{N,T}(R)\bigr)\,\lambda_{N,T}(dR).
\]

Hence for every bounded measurable gauge observable \(\Phi\) on \(\mathscr X_{N,T}\),

\[
\left|\int\Phi\,d\mu_{N,T}-\int\Phi\,d\mu_W^{(N,T)}\right|
\le
2\|\Phi\|_\infty
\int_{\mathscr R_{N,T}} \sinh\bigl(\varepsilon_{N,T}(R)\bigr)\,\lambda_{N,T}(dR).
\]

#### Proof

Disintegrate both laws over the common regulator law \(\lambda_{N,T}\) and integrate the fiberwise total-variation bound from Theorem 12.2. The observable estimate follows by the same total-variation inequality. ∎

### Corollary 12.4 (asymptotic Wilson equivalence criterion)

Let \((N_k,T_k)\) be a sequence of finite regulators. If

\[
\int_{\mathscr R_{N_k,T_k}} \sinh\bigl(\varepsilon_{N_k,T_k}(R)\bigr)\,\lambda_{N_k,T_k}(dR)
\longrightarrow 0,
\]

then

\[
\|\mu_{N_k,T_k}-\mu_W^{(N_k,T_k)}\|_{\mathrm{TV}}\to 0.
\]

In particular, every bounded gauge observable has the same limit under the induced law and the mixed Wilson law.

#### Proof

Immediate from Corollary 12.3. ∎

---

## 13. What Stage I now proves

After inserting the package above into `03b`, the paper will have the following theorem structure.

### Unconditional finite-volume theorem package

1. The realized-regulator data space is standard Borel.
2. The finite gauge fibers are compact Polish spaces.
3. The pushforward induced gauge law exists on the total gauge space.
4. The induced gauge law disintegrates over realized regulators.
5. Wilson loops and the Wilson action are bounded measurable observables.

### Conditional finite-volume Yang--Mills-type theorem package

6. Under the gauge-equivariant lift assumption, the conditional fiber laws are gauge-invariant.
7. Under fiberwise absolute continuity, the exact effective action exists.
8. Under the Wilson comparison remainder assumption, the induced fiber law is exponentially close to the Wilson fiber law in total variation.
9. If the remainder goes to zero under refinement, the induced finite gauge theory becomes asymptotically equivalent to a mixed Wilson gauge theory.

This is the correct finite-volume measure-theoretic base for the later Euclidean QFT stages.

---

## 14. Exact insertion plan for `03b`

The following changes should be made to the revised `03b` manuscript.

### Replace the current single theorem on the induced gauge law

The current theorem that only states existence of the pushforward measure should be replaced by the following staged package:

- **Definition:** realized regulator-data space \(\mathscr R_{N,T}\)
- **Definition:** fiber gauge space \(\mathscr U(R)\)
- **Definition:** total gauge space \(\mathscr X_{N,T}\)
- **Theorem:** induced law and disintegration (Theorems 6.2 and 6.4 above)
- **Definition:** fiber Haar measure \(\nu_R\)
- **Assumption:** fiberwise absolute continuity
- **Theorem:** exact effective action (Theorem 10.2)
- **Definition:** fiber Wilson measure \(\mu_W^R\)
- **Assumption:** Wilson comparison remainder
- **Theorem:** fiberwise Wilson comparison (Theorem 12.2)
- **Corollary:** mixed-law comparison and asymptotic equivalence (Corollaries 12.3 and 12.4)

### Keep the current Wilson action section, but reinterpret it

The existing Wilson-action definitions and small-loop formulas should now be read as defining the **reference Wilson fiber measure** \(\mu_W^R\), not as the final finite gauge law of the theory.

### Move the continuum comparison theorem later

The continuum comparison theorem should remain in `03b`, but its role is now strictly secondary:

- first: define and control the finite gauge law,
- later: compare its local small-loop expectations to the classical continuum field.

This ordering is essential.

---

## 15. What remains to be proved from the actual Fractal-Gas dynamics

Stage I is rigorous as written, but two assumptions still have to be discharged from the model.

## 15.1 To prove fiberwise absolute continuity (Assumption 10.1)

You need a model-specific argument that, conditional on a realized regulator, the induced edge variables have a density relative to product Haar measure.

Possible routes:

1. **Add explicit edge noise in Lie algebra coordinates** before exponentiation; then absolute continuity is often immediate.
2. **Prove smooth nondegeneracy of the edge phase/color map** from the swarm noise to the edge transports.
3. **Use Malliavin-type arguments** if the edge transports are smooth functionals of nondegenerate Gaussian driving noise.

Without one of these, an induced pushforward measure may remain singular.

## 15.2 To prove the Wilson comparison remainder (Assumption 12.1)

You need to identify the exact effective action and then prove it is close to the Wilson action.

The natural model-side route is:

1. expand the conditional log-density of the edge transports,
2. isolate the plaquette-local quadratic contribution,
3. identify that contribution with the Wilson action,
4. prove the remainder is gauge-invariant and has small oscillation.

Technically, this is where local cumulant control, small-loop expansions, and conditional independence / mixing estimates enter.

A sufficient strategy is to show that, conditional on the realized regulator, the edge law is a **quasi-local Gibbs measure** whose interaction potential has:

- leading plaquette term = Wilson action,
- all higher interactions uniformly small.

Then the oscillation estimate on \(\Delta_{N,T}\) follows.

---

## 16. Why this Stage I package is the correct target

This package is the right foundation because it solves the exact problem that the current revised `03b` still leaves open.

Before Stage I, you have:

- a random regulator,
- random gauge variables,
- Wilson loops,
- and a pushforward law.

After Stage I, you have:

- a **finite-volume gauge measure theory** on random regulators,
- exact conditional gauge measures,
- an exact effective action under a clean absolute-continuity hypothesis,
- and a rigorous comparison theorem showing when the swarm-induced law is actually Wilson-like.

That is the correct launching point for the later Euclidean-QFT and mass-gap stages.

---

## 17. Final summary

If you adopt this document as the Stage I replacement in `03b`, then the paper will no longer stop at:

> "there exists a pushforward gauge law."

It will instead prove the stronger and structurally correct statement:

> the realized Fractal regulator supports a finite-volume gauge measure theory, disintegrated over realized regulators, with an exact effective action and a rigorous criterion for Wilson equivalence.

That is the measure-theoretic base you need before you can honestly build a fully fledged Euclidean Yang--Mills QFT on top of the Fractal-Gas construction.
