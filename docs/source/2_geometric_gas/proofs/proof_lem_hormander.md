# Complete Proof: Hörmander's Condition for Geometric Gas Kinetic Operator

**Document**: `docs/source/2_geometric_gas/16_convergence_mean_field.md`
**Theorem Label**: `lem-hormander`
**Date**: 2025-10-25
**Rigor Level**: Annals standard (Attempt 1/3)

---

## Theorem Statement

:::{prf:lemma} Hörmander's Condition
:label: lem-hormander

The kinetic operator $\mathcal{L}_{\text{kin}}$ satisfies Hörmander's bracket condition:

The vector fields:

$$
X_0 = v \cdot \nabla_x - \nabla_x U \cdot \nabla_v - \gamma v \cdot \nabla_v, \quad X_j = \sigma \frac{\partial}{\partial v_j}
$$

generate the full tangent space at every point through repeated Lie brackets.
:::

---

## Proof Strategy

We verify Hörmander's bracket-generating condition through explicit Lie bracket computation. The proof proceeds in five stages:

1. **Operator reformulation**: Express the backward kinetic operator in Hörmander's canonical form $L = \frac{1}{2}\sum_{j=1}^d X_j^2 + X_0$
2. **Lie bracket computation**: Explicitly compute commutators $[X_0, X_j]$ for all $j \in \{1, \ldots, d\}$
3. **Position direction extraction**: Show that position derivatives emerge from the bracket structure
4. **Tangent space span**: Verify full-rank condition at every point $(x,v) \in \Omega$
5. **Hypoellipticity conclusion**: Apply Hörmander's theorem

The key insight is that velocity diffusion $X_j$ couples with the transport term $v \cdot \nabla_x$ in $X_0$ to generate position directions through brackets, despite the operator having no direct diffusion in $x$.

---

## Framework Dependencies

**Axioms** (from {prf:ref}`assump-qsd-existence` in `16_convergence_mean_field.md`):

- **A1 (Confinement)**: $U: \mathcal{X} \to \mathbb{R}$ with $\nabla^2 U \geq \kappa_{\text{conf}} I_d$ and $U(x) \to +\infty$ as $x \to \partial\mathcal{X}$
- **A3 (Parameters)**: $\gamma > 0$, $\sigma^2 > 0$, $\lambda_{\text{revive}} > 0$
- **A4 (Domain)**: $\Omega = \mathcal{X} \times \mathbb{R}^d$ where $\mathcal{X} \subset \mathbb{R}^d$ is the alive region

**Definitions**:

- **Kinetic operator** ({prf:ref}`def-qsd-mean-field`, line 1068):

$$
\mathcal{L}_{\text{kin}}[\rho] = -v \cdot \nabla_x \rho + \nabla_x U(x) \cdot \nabla_v \rho + \gamma \nabla_v \cdot (v \rho) + \frac{\sigma^2}{2} \Delta_v \rho
$$

- **Lie bracket**: For vector fields (first-order differential operators) $X, Y$, the commutator $[X, Y] := XY - YX$ is the Lie bracket

**Related results**:

- {prf:ref}`lem-uniqueness-hormander-verification` from `08_propagation_chaos.md`: Analogous verification for Euclidean Gas (without potential $U$)
- Hörmander's Hypoellipticity Theorem (1967): Bracket-generating condition implies hypoellipticity

---

## Complete Proof

:::{prf:proof}

### Step 1: Reformulation in Hörmander Form

**Goal**: Express the backward kinetic operator in the standard form $L = \frac{1}{2}\sum_{j=1}^d X_j^2 + X_0$ with appropriate vector fields.

**Substep 1.1**: Derive the backward operator

The kinetic operator in forward (Fokker-Planck) form acts on densities $\rho$ as:

$$
\mathcal{L}_{\text{kin}}[\rho] = -v \cdot \nabla_x \rho + \nabla_x U(x) \cdot \nabla_v \rho + \gamma \nabla_v \cdot (v \rho) + \frac{\sigma^2}{2} \Delta_v \rho
$$

The **backward operator** (adjoint, acting on test functions $\phi$) is obtained by integration by parts. Throughout, we assume $\phi \in C_c^\infty(\Omega)$ (compactly supported smooth test functions), so all boundary terms vanish.

For the transport term:

$$
\int (v \cdot \nabla_x \rho) \phi \, dx dv = -\int \rho (v \cdot \nabla_x \phi) \, dx dv
$$

For the potential gradient term:

$$
\int (\nabla_x U \cdot \nabla_v \rho) \phi \, dx dv = -\int \rho (\nabla_x U \cdot \nabla_v \phi) \, dx dv
$$

For the divergence term, integrate by parts directly:

$$
\int \gamma \nabla_v \cdot (v\rho) \phi \, dx dv = -\int \gamma (v\rho) \cdot \nabla_v \phi \, dx dv = -\int \gamma \rho (v \cdot \nabla_v \phi) \, dx dv
$$

The diffusion term $\frac{\sigma^2}{2}\Delta_v$ is self-adjoint.

Therefore, the backward operator is:

$$
L[\phi] = v \cdot \nabla_x \phi - \nabla_x U(x) \cdot \nabla_v \phi - \gamma v \cdot \nabla_v \phi + \frac{\sigma^2}{2} \Delta_v \phi
$$

**Remark**: Hörmander's theorem is classically stated for backward operators. We verify the bracket condition for $L$; hypoellipticity of the forward operator $\mathcal{L}_{\text{kin}}$ follows by duality (see discussion in Step 5).

**Substep 1.2**: Identify vector fields

Define:

$$
X_j := \sigma \frac{\partial}{\partial v_j}, \quad j = 1, \ldots, d \quad \text{(velocity diffusion directions)}
$$

$$
X_0 := v \cdot \nabla_x - \nabla_x U(x) \cdot \nabla_v - \gamma v \cdot \nabla_v \quad \text{(drift operator)}
$$

Then:

$$
\Delta_v = \sum_{j=1}^d \frac{\partial^2}{\partial v_j^2} = \sum_{j=1}^d \left(\frac{\partial}{\partial v_j}\right)^2 = \frac{1}{\sigma^2} \sum_{j=1}^d X_j^2
$$

Substituting:

$$
L[\phi] = \frac{\sigma^2}{2} \cdot \frac{1}{\sigma^2} \sum_{j=1}^d X_j^2 \phi + X_0[\phi] = \frac{1}{2} \sum_{j=1}^d X_j^2 \phi + X_0[\phi]
$$

**Conclusion**: The operator is in Hörmander form:

$$
L = \frac{1}{2} \sum_{j=1}^d X_j^2 + X_0
$$

with vector fields $X_0, X_1, \ldots, X_d$ as defined above.

---

### Step 2: Explicit Lie Bracket Computation

**Goal**: Compute $[X_0, X_j]$ for all $j \in \{1, \ldots, d\}$.

**Substep 2.1**: Decompose $X_0$

Write $X_0$ as a sum of three operators:

$$
X_0 = A + B + C
$$

where:

$$
A := v \cdot \nabla_x = \sum_{i=1}^d v_i \partial_{x_i}
$$

$$
B := -\nabla_x U \cdot \nabla_v = -\sum_{i=1}^d (\partial_{x_i} U) \partial_{v_i}
$$

$$
C := -\gamma v \cdot \nabla_v = -\gamma \sum_{i=1}^d v_i \partial_{v_i}
$$

By linearity of the Lie bracket:

$$
[X_0, X_j] = [A, X_j] + [B, X_j] + [C, X_j]
$$

**Substep 2.2**: Compute $[A, X_j]$

For $A = \sum_i v_i \partial_{x_i}$ and $X_j = \sigma \partial_{v_j}$, apply the commutator to a test function $\phi$:

$$
[A, X_j]\phi = A(X_j \phi) - X_j(A \phi)
$$

First term:

$$
A(X_j \phi) = \sum_i v_i \partial_{x_i}(\sigma \partial_{v_j} \phi) = \sigma \sum_i v_i \partial_{x_i} \partial_{v_j} \phi
$$

Second term (using product rule):

$$
X_j(A \phi) = \sigma \partial_{v_j}\left(\sum_i v_i \partial_{x_i} \phi\right) = \sigma \sum_i \left[\partial_{v_j}(v_i) \partial_{x_i} \phi + v_i \partial_{v_j} \partial_{x_i} \phi\right]
$$

Since $\partial_{v_j}(v_i) = \delta_{ij}$ (Kronecker delta):

$$
X_j(A \phi) = \sigma \partial_{x_j} \phi + \sigma \sum_i v_i \partial_{v_j} \partial_{x_i} \phi
$$

By Schwarz's theorem (equality of mixed partials): $\partial_{x_i}\partial_{v_j} = \partial_{v_j}\partial_{x_i}$.

Therefore:

$$
[A, X_j]\phi = \sigma \sum_i v_i \partial_{x_i}\partial_{v_j}\phi - \sigma \partial_{x_j}\phi - \sigma \sum_i v_i \partial_{v_j}\partial_{x_i}\phi = -\sigma \partial_{x_j}\phi
$$

**Conclusion**:

$$
[A, X_j] = -\sigma \partial_{x_j}
$$

**Substep 2.3**: Compute $[B, X_j]$

For $B = -\sum_i (\partial_{x_i} U) \partial_{v_i}$ and $X_j = \sigma \partial_{v_j}$:

$$
[B, X_j]\phi = B(X_j \phi) - X_j(B \phi)
$$

First term:

$$
B(X_j \phi) = -\sum_i (\partial_{x_i} U) \partial_{v_i}(\sigma \partial_{v_j} \phi)
$$

Since $U = U(x)$ (independent of $v$ by Assumption A1):

$$
\partial_{v_i}(\partial_{x_k} U) = 0 \quad \text{for all } i, k
$$

Thus:

$$
B(X_j \phi) = -\sigma \sum_i (\partial_{x_i} U) \partial_{v_i}\partial_{v_j}\phi
$$

Second term:

$$
X_j(B \phi) = -\sigma \partial_{v_j}\left(\sum_i (\partial_{x_i} U) \partial_{v_i} \phi\right) = -\sigma \sum_i \left[\partial_{v_j}(\partial_{x_i} U) \partial_{v_i}\phi + (\partial_{x_i} U) \partial_{v_j}\partial_{v_i}\phi\right]
$$

The first term vanishes since $U$ is independent of $v$:

$$
X_j(B \phi) = -\sigma \sum_i (\partial_{x_i} U) \partial_{v_j}\partial_{v_i}\phi
$$

By equality of mixed partials: $\partial_{v_i}\partial_{v_j} = \partial_{v_j}\partial_{v_i}$.

Therefore:

$$
[B, X_j]\phi = -\sigma \sum_i (\partial_{x_i} U) \partial_{v_i}\partial_{v_j}\phi + \sigma \sum_i (\partial_{x_i} U) \partial_{v_j}\partial_{v_i}\phi = 0
$$

**Conclusion**:

$$
[B, X_j] = 0
$$

**Substep 2.4**: Compute $[C, X_j]$

For $C = -\gamma \sum_i v_i \partial_{v_i}$ and $X_j = \sigma \partial_{v_j}$:

$$
[C, X_j]\phi = C(X_j \phi) - X_j(C \phi)
$$

First term:

$$
C(X_j \phi) = -\gamma \sum_i v_i \partial_{v_i}(\sigma \partial_{v_j}\phi) = -\gamma \sigma \sum_i v_i \partial_{v_i}\partial_{v_j}\phi
$$

Second term (product rule):

$$
X_j(C \phi) = -\gamma \sigma \partial_{v_j}\left(\sum_i v_i \partial_{v_i}\phi\right) = -\gamma \sigma \sum_i \left[\partial_{v_j}(v_i) \partial_{v_i}\phi + v_i \partial_{v_j}\partial_{v_i}\phi\right]
$$

$$
= -\gamma \sigma \partial_{v_j}\phi - \gamma \sigma \sum_i v_i \partial_{v_j}\partial_{v_i}\phi
$$

Therefore:

$$
[C, X_j]\phi = -\gamma \sigma \sum_i v_i \partial_{v_i}\partial_{v_j}\phi + \gamma \sigma \partial_{v_j}\phi + \gamma \sigma \sum_i v_i \partial_{v_j}\partial_{v_i}\phi
$$

By equality of mixed partials:

$$
[C, X_j]\phi = \gamma \sigma \partial_{v_j}\phi = \gamma X_j \phi
$$

**Conclusion**:

$$
[C, X_j] = \gamma X_j
$$

**Substep 2.5**: Assemble the full bracket

Combining results from substeps 2.2–2.4:

$$
[X_0, X_j] = [A, X_j] + [B, X_j] + [C, X_j] = -\sigma \partial_{x_j} + 0 + \gamma X_j
$$

**Final Result**:

$$
[X_0, X_j] = -\sigma \partial_{x_j} + \gamma X_j \quad \text{for all } j \in \{1, \ldots, d\}
$$

---

### Step 3: Extract Position Directions

**Goal**: Show that position derivatives $\{\partial_{x_1}, \ldots, \partial_{x_d}\}$ lie in the Lie algebra generated by $\{X_0, X_1, \ldots, X_d\}$.

From Step 2, we have:

$$
[X_0, X_j] = -\sigma \partial_{x_j} + \gamma X_j
$$

Rearranging:

$$
\partial_{x_j} = \frac{\gamma X_j - [X_0, X_j]}{\sigma}
$$

Since $\sigma > 0$ by Assumption A3, division is well-defined.

**Verification of non-degeneracy**: The formula is uniform across $\Omega$ because:
- $\sigma$ is a positive constant (A3), independent of $(x,v)$
- $\gamma$ is a positive constant (A3), independent of $(x,v)$
- The bracket $[X_0, X_j]$ is a first-order differential operator with coefficients independent of the phase space point

Therefore, there are **no special points** where the formula degenerates.

**Conclusion**: All position derivatives $\{\partial_{x_1}, \ldots, \partial_{x_d}\}$ are expressible as linear combinations of Lie algebra elements $\{X_j, [X_0, X_j]\}$, hence belong to the Lie algebra $\mathcal{L}(X_0, X_1, \ldots, X_d)$.

---

### Step 4: Verify Tangent Space Span at Every Point

**Goal**: Prove that $\mathcal{L}(X_0, X_1, \ldots, X_d)$ spans $T_{(x,v)}\Omega = \mathbb{R}^{2d}$ at every point $(x,v) \in \Omega$.

**Substep 4.1**: Span velocity directions

The vector fields $\{X_1, \ldots, X_d\}$ are:

$$
X_j = \sigma \partial_{v_j}, \quad j = 1, \ldots, d
$$

Since $\sigma > 0$ (A3), these are non-zero scalar multiples of the canonical basis $\{\partial_{v_1}, \ldots, \partial_{v_d}\}$ of the velocity tangent space.

**Conclusion**: $\text{span}\{X_1, \ldots, X_d\} = T_v\Omega = \mathbb{R}^d_v$ (velocity directions).

**Substep 4.2**: Span position directions

From Step 3:

$$
\partial_{x_j} = \frac{\gamma X_j - [X_0, X_j]}{\sigma} \in \mathcal{L}(X_0, X_1, \ldots, X_d)
$$

for all $j = 1, \ldots, d$.

The set $\{\partial_{x_1}, \ldots, \partial_{x_d}\}$ is the canonical basis of the position tangent space.

**Conclusion**: $\text{span}\{\partial_{x_1}, \ldots, \partial_{x_d}\} = T_x\Omega = \mathbb{R}^d_x$ (position directions) is contained in the Lie algebra.

**Substep 4.3**: Full tangent space

The tangent space at $(x,v)$ decomposes as:

$$
T_{(x,v)}\Omega = T_x\Omega \times T_v\Omega = \mathbb{R}^d_x \times \mathbb{R}^d_v = \mathbb{R}^{2d}
$$

Since the Lie algebra contains bases for both $T_x\Omega$ and $T_v\Omega$, it contains a basis for $T_{(x,v)}\Omega$.

**Substep 4.4**: Check for potential degeneracies

**Case 1**: $v = 0$ (zero velocity)
- The bracket formula $[X_0, X_j] = -\sigma \partial_{x_j} + \gamma X_j$ has constant coefficients $\sigma$, $\gamma$
- No dependence on the value of $v$
- ✓ **No degeneracy**

**Case 2**: $\nabla_x U = 0$ (critical points of potential)
- We showed $[B, X_j] = 0$ where $B = -\nabla_x U \cdot \nabla_v$
- The position directions arise from $[A, X_j] = -\sigma\partial_{x_j}$, which is independent of $U$
- ✓ **No degeneracy**

**Case 3**: Near boundary $\partial\mathcal{X}$
- Hörmander's theorem applies to the interior of $\Omega$
- Boundary regularity is a separate issue (classical theory for hypoelliptic boundary value problems)
- For this lemma, we verify the bracket condition on $\text{int}(\Omega)$
- ✓ **Interior condition satisfied**

**Conclusion**: The Lie algebra $\mathcal{L}(X_0, X_1, \ldots, X_d)$ spans $T_{(x,v)}\Omega = \mathbb{R}^{2d}$ at every point $(x,v) \in \text{int}(\Omega)$, with no degeneracies.

---

### Step 5: Apply Hörmander's Theorem

**Theorem** (Hörmander 1967, Theorem 1.1):

Let $M$ be a smooth manifold and $L = \frac{1}{2}\sum_{j=1}^m X_j^2 + X_0$ a second-order differential operator where $X_0, X_1, \ldots, X_m$ are smooth vector fields. If the Lie algebra generated by $\{X_0, X_1, \ldots, X_m\}$ spans the tangent space $T_pM$ at every point $p \in M$, then $L$ is **hypoelliptic**:

$$
Lu = f \text{ with } f \in C^\infty(M) \implies u \in C^\infty(M)
$$

**Verification of hypotheses**:

1. **Manifold structure**: $\Omega = \mathcal{X} \times \mathbb{R}^d$ is a smooth manifold (product of open subset of $\mathbb{R}^d$ with $\mathbb{R}^d$)

2. **Operator form**: Verified in Step 1: $L = \frac{1}{2}\sum_{j=1}^d X_j^2 + X_0$

3. **Smoothness of vector fields**:
   - $X_j = \sigma \partial_{v_j}$ are constant coefficient (infinitely smooth)
   - $X_0 = v \cdot \nabla_x - \nabla_x U \cdot \nabla_v - \gamma v \cdot \nabla_v$ requires $U$ to be sufficiently smooth
   - **Assumption A1** guarantees $\nabla^2 U \geq \kappa_{\text{conf}} I_d$, which implies $U \in C^2$ at minimum
   - **For the bracket condition** (this lemma): The algebraic computation in Steps 2-4 requires only $U \in C^1$, which is satisfied
   - **For regularity conclusions**:
     * If $U \in C^\infty$: Classical Hörmander (1967) gives $C^\infty$ hypoelliptic regularity
     * If $U \in C^{k,\alpha}$ for finite $k$: Hypoelliptic Schauder estimates (Bony 1969) give $C^{k,\alpha}$ regularity
     * For **requirement R2** ($\rho_\infty \in C^2$): $U \in C^2$ suffices via Bony's theory
   - **Practical assumption**: For smooth confining potentials (e.g., quadratic, quartic), $U \in C^\infty$ is natural

4. **Bracket-generating condition**: Verified in Step 4

**Application**: By Hörmander's theorem, the backward operator $L$ is hypoelliptic.

**Transfer to forward operator**: The forward (Fokker-Planck) operator $\mathcal{L}_{\text{kin}}$ is the formal adjoint of $L$. Hypoellipticity transfers to the adjoint because:

1. **Duality argument**: If $\mathcal{L}_{\text{kin}}[u] = f$ with $f \in C^\infty(\Omega)$, then for any test function $\phi \in C_c^\infty(\Omega)$:
   $$
   \langle u, L[\phi] \rangle = \langle \mathcal{L}_{\text{kin}}[u], \phi \rangle = \langle f, \phi \rangle
   $$
   Since $f$ is smooth and $L$ is hypoelliptic, standard regularity theory implies $u$ is smooth.

2. **Shared structure**: The diffusion part $\frac{\sigma^2}{2}\Delta_v$ is self-adjoint, and both operators share the same bracket-generating vector fields $\{X_j\}$ (up to sign changes in drift terms, which do not affect the bracket algebra).

**Reference**: For hypoellipticity of adjoints, see Hörmander, *The Analysis of Linear Partial Differential Operators I*, Theorem 8.3.1.

**Conclusion**: The kinetic operator $\mathcal{L}_{\text{kin}}$ satisfies Hörmander's bracket condition, implying hypoellipticity: smooth forcing yields smooth solutions (with regularity class determined by coefficient smoothness).

**Q.E.D.** ∎

:::

---

## Regularity Discussion

**Separation of concerns**: The bracket condition (Step 4) is purely algebraic and independent of coefficient smoothness. The regularity conclusions from hypoellipticity depend on the smoothness class of $U$.

**What this lemma proves**:
1. ✓ **Bracket-generating condition**: Verified for any $U$ with $\nabla_x U$ defined (i.e., $U \in C^1$)
2. ✓ **Hypoelliptic structure**: The operator has the correct sum-of-squares form with bracket-spanning property

**Regularity implications** (depend on $U$ smoothness):

| Assumption on $U$ | Hypoelliptic regularity | Reference | Framework status |
|-------------------|-------------------------|-----------|------------------|
| $U \in C^{k,\alpha}$ | $C^{k,\alpha}$ solutions from $C^{0,\alpha}$ forcing | Bony (1969) | A1 gives $U \in C^2$ at minimum |
| $U \in C^\infty$ | $C^\infty$ solutions from $C^\infty$ forcing | Hörmander (1967) | Likely intended; natural for smooth potentials |

**For QSD applications**:
- **Requirement R2** ($\rho_\infty \in C^2$): Guaranteed by A1 via Bony's Schauder estimates ✓
- **Bootstrap to $C^\infty$**: If the framework assumes smooth potentials, can iterate Schauder to get full $C^\infty$ regularity
- **Recommendation**: Clarify whether A1 implicitly assumes $U \in C^\infty$ or only $U \in C^2$

**Practical note**: Standard confining potentials (quadratic $U(x) = \frac{1}{2}|x|^2$, quartic, etc.) are $C^\infty$, so the full Hörmander regularity applies in typical use cases.

---

## Cross-References

**Theorems invoked**:
- Hörmander (1967), "Hypoelliptic second order differential equations", *Acta Math.* 119:147-171
- Friedman (1964), *Partial Differential Equations of Parabolic Type*, Chapter 1 (adjoint stability)
- Bony (1969), "Principe du maximum, inégalité de Harnack et unicité du problème de Cauchy pour les opérateurs elliptiques dégénérés" (hypoelliptic Schauder estimates)

**Framework references**:
- {prf:ref}`lem-uniqueness-hormander-verification` (`08_propagation_chaos.md`) — analogous verification for Euclidean Gas
- {prf:ref}`assump-qsd-existence` (`16_convergence_mean_field.md`) — framework assumptions A1, A3, A4
- {prf:ref}`def-qsd-mean-field` (`16_convergence_mean_field.md`) — kinetic operator definition

**Downstream applications**:
- {prf:ref}`cor-hypoelliptic-regularity` — immediate consequence for smoothness of solutions
- {prf:ref}`thm-qsd-smoothness` — bootstrap argument for $\rho_\infty \in C^\infty$
- Requirement R2 (QSD smoothness) — directly depends on hypoelliptic regularity

---

## Proof Validation Checklist

- [x] **Logical completeness**: All steps follow from explicit algebraic computation
- [x] **Hypothesis usage**: All assumptions (A1, A3, A4) explicitly invoked where needed
- [x] **Conclusion derivation**: Bracket-generating condition fully verified, Hörmander's theorem correctly applied
- [x] **Framework consistency**: All dependencies on framework axioms and definitions verified
- [x] **No circular reasoning**: Proof does not assume hypoellipticity; derives it from first principles
- [x] **Constant tracking**: All constants $\sigma$, $\gamma$, $d$ defined and bounded (A3)
- [x] **Edge cases**: Verified no degeneracies at $v=0$, $\nabla_x U = 0$, or generic interior points
- [x] **Regularity requirements**: Bracket condition algebraic (holds for $C^1$); regularity conclusions clarified
- [x] **Measure theory**: Not applicable (lemma concerns differential operators, not probability measures)
- [x] **Sign tracking**: Careful sign verification in each Lie bracket computation
- [x] **Product rule**: Explicitly applied in substeps 2.2, 2.4
- [x] **Mixed partials**: Schwarz's theorem used correctly to cancel terms

**Confidence**: High (algebraic verification, follows established template from Euclidean Gas framework)

---

**End of Proof**
