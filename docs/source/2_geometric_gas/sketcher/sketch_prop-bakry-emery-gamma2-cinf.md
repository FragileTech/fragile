# Proof Sketch for prop-bakry-emery-gamma2-cinf

**Document**: docs/source/2_geometric_gas/19_geometric_gas_cinf_regularity_simplified.md
**Theorem**: prop-bakry-emery-gamma2-cinf
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:proposition} Bakry-Émery Curvature Condition (Conditional)
:label: prop-bakry-emery-gamma2-cinf

Assume V_fit ∈ C∞ and V_total uniformly convex with ∇²V_total ≥ λ_BE I. Then:

$$
\Gamma_2(f, f) \geq \lambda_{\text{BE}} \Gamma(f, f) \quad \forall \text{ smooth } f
$$

where:
- Γ(f, g) = (1/2)[L(fg) - f Lg - g Lf] = ⟨∇f, ∇g⟩ (carré du champ)
- Γ_2(f, f) = (1/2)[L Γ(f, f) - 2 Γ(f, Lf)] (iterated carré du champ)

**Implication**: Γ_2 ≥ λΓ implies:
- Spectral gap λ_gap ≥ λ_BE
- LSI with constant 1/λ_BE
- Hypercontractivity (Nelson's theorem)

**Proof**: Computing Γ_2 requires third and fourth derivatives of V_total. C∞ regularity ensures all terms well-defined. Uniform convexity yields lower bound. See Bakry & Émery, *Séminaire de Probabilités XIX* (1985). □
:::

**Informal Restatement**: The Bakry-Émery Γ_2 criterion is a powerful **curvature condition** for diffusion operators that guarantees exponential convergence, functional inequalities, and hypercontractivity. It states that the "iterated carré du champ" operator Γ_2 (a second-order differential operator acting on the square of gradients) is bounded below by a positive constant times the "carré du champ" operator Γ (the square of gradients).

**Physical Interpretation**: The Γ_2 ≥ λ Γ condition is a **Ricci curvature lower bound** in the sense of Bakry and Émery. For a Riemannian manifold with metric induced by the generator L, this condition says the Ricci curvature is bounded below by λ. In the context of diffusion processes, it means the drift term (gradient of potential) provides sufficient "restoring force" to prevent trajectories from escaping to infinity, ensuring exponential relaxation to equilibrium.

**Conditional Nature**: This proposition is marked as "Conditional" because it assumes:
1. **V_fit ∈ C∞**: Established in this document (Theorem {prf:ref}`thm-cinf-regularity`)
2. **Uniform convexity**: ∇²V_total ≥ λ_BE I - This is **NOT** generally satisfied by V_fit
   - V_fit depends on empirical diversity, not uniformly convex in position
   - May hold for specific choices of confining potential U and small ε_F
   - Needs case-by-case verification

---

## II. Proof Strategy

### Chosen Method: Direct Computation of Γ_2 via Bakry-Émery Calculus

**High-Level Approach**:
The proof follows the standard **Bakry-Émery calculus** for proving the Γ_2 criterion:
1. Define the generator L and compute the carré du champ operator Γ
2. Compute the iterated carré du champ Γ_2 using commutator identities
3. Use uniform convexity of V_total to bound Γ_2 from below
4. Identify the curvature constant λ_BE in terms of the Hessian lower bound

**Key Insight**: For generators of the form L = Δ - ∇V · ∇, the Γ_2 operator can be computed explicitly using the Bochner-Weitzenböck formula:

$$
\Gamma_2(f, f) = \|\nabla^2 f\|^2_{HS} + \langle \nabla f, \nabla^2 V \nabla f \rangle
$$

When V is uniformly convex (∇²V ≥ λI), the second term provides the lower bound.

---

### Detailed Proof Steps

**Step 1: Define the Generator and Carré du Champ**

The infinitesimal generator of the diffusion process with regularized noise is:

$$
\mathcal{L} f = \frac{1}{2} \text{Tr}(\Sigma_{\text{reg}}^2 \nabla^2 f) - \nabla V_{\text{total}} \cdot \nabla f
$$

where:
- V_total = U + ε_F V_fit is the total potential
- Σ_reg is the regularized diffusion tensor (uniformly elliptic)

**For simplicity**, assume Σ_reg = I (standard diffusion). The general case follows by change of variables.

The carré du champ operator is:

$$
\Gamma(f, g) := \frac{1}{2}[\mathcal{L}(fg) - f \mathcal{L}g - g \mathcal{L}f]
$$

**Computation**:

$$
\mathcal{L}(fg) = \frac{1}{2}\Delta(fg) - \nabla V_{\text{total}} \cdot \nabla(fg)
$$

$$
= \frac{1}{2}[f \Delta g + g \Delta f + 2 \nabla f \cdot \nabla g] - f \nabla V_{\text{total}} \cdot \nabla g - g \nabla V_{\text{total}} \cdot \nabla f
$$

Substituting $\mathcal{L}f = \frac{1}{2}\Delta f - \nabla V_{\text{total}} \cdot \nabla f$ and $\mathcal{L}g$ similarly:

$$
\Gamma(f, g) = \frac{1}{2} \cdot 2 \nabla f \cdot \nabla g = \langle \nabla f, \nabla g \rangle
$$

**Result**: For the standard Langevin generator, Γ(f, g) = ⟨∇f, ∇g⟩ is the standard Dirichlet form.

---

**Step 2: Compute the Iterated Carré du Champ Γ_2**

The iterated carré du champ is:

$$
\Gamma_2(f, f) := \frac{1}{2}[\mathcal{L}\Gamma(f, f) - 2\Gamma(f, \mathcal{L}f)]
$$

Since Γ(f, f) = |∇f|², we have:

$$
\Gamma_2(f, f) = \frac{1}{2}\left[\mathcal{L}|\nabla f|^2 - 2\Gamma(f, \mathcal{L}f)\right]
$$

**Step 2a: Compute L|∇f|²**

$$
\mathcal{L}|\nabla f|^2 = \frac{1}{2}\Delta|\nabla f|^2 - \nabla V_{\text{total}} \cdot \nabla|\nabla f|^2
$$

Using the product rule and commutation identities:

$$
\Delta|\nabla f|^2 = 2|\nabla^2 f|^2_{HS} + 2\nabla f \cdot \nabla(\Delta f)
$$

where $\|\nabla^2 f\|^2_{HS} := \sum_{i,j} (\partial_i \partial_j f)^2$ is the Hilbert-Schmidt norm.

$$
\nabla|\nabla f|^2 = 2\nabla^2 f \nabla f
$$

**Substitute**:

$$
\mathcal{L}|\nabla f|^2 = |\nabla^2 f|^2_{HS} + \nabla f \cdot \nabla(\Delta f) - \nabla V_{\text{total}} \cdot (\nabla^2 f \nabla f)
$$

**Step 2b: Compute Γ(f, Lf)**

$$
\Gamma(f, \mathcal{L}f) = \langle \nabla f, \nabla \mathcal{L}f \rangle
$$

$$
\nabla \mathcal{L}f = \frac{1}{2}\nabla(\Delta f) - \nabla(\nabla V_{\text{total}} \cdot \nabla f)
$$

$$
= \frac{1}{2}\nabla(\Delta f) - \nabla^2 V_{\text{total}} \nabla f - \nabla V_{\text{total}} \cdot \nabla^2 f
$$

Wait, the last term should use the Hessian:

$$
\nabla(\nabla V_{\text{total}} \cdot \nabla f) = (\nabla^2 V_{\text{total}}) \nabla f + \nabla V_{\text{total}} \cdot \nabla^2 f
$$

where the second term is $\nabla[\nabla V_{\text{total}} \cdot \nabla f]$ via the product rule.

Actually, let me use index notation for clarity:

$$
\partial_i(\partial_j V_{\text{total}} \partial_j f) = (\partial_i \partial_j V_{\text{total}}) \partial_j f + \partial_j V_{\text{total}} \partial_i \partial_j f
$$

So:

$$
\nabla(\nabla V_{\text{total}} \cdot \nabla f) = (\nabla^2 V_{\text{total}}) \nabla f + \nabla^2 f \nabla V_{\text{total}}
$$

where the second term is a matrix-vector product (Hessian of f times gradient of V).

Therefore:

$$
\nabla \mathcal{L}f = \frac{1}{2}\nabla(\Delta f) - (\nabla^2 V_{\text{total}}) \nabla f - (\nabla^2 f) \nabla V_{\text{total}}
$$

And:

$$
\Gamma(f, \mathcal{L}f) = \nabla f \cdot \left[\frac{1}{2}\nabla(\Delta f) - (\nabla^2 V_{\text{total}}) \nabla f - (\nabla^2 f) \nabla V_{\text{total}}\right]
$$

$$
= \frac{1}{2}\nabla f \cdot \nabla(\Delta f) - \langle \nabla f, \nabla^2 V_{\text{total}} \nabla f \rangle - \nabla f \cdot (\nabla^2 f) \nabla V_{\text{total}}
$$

**Step 2c: Assemble Γ_2**

$$
\Gamma_2(f, f) = \frac{1}{2}\left[\mathcal{L}|\nabla f|^2 - 2\Gamma(f, \mathcal{L}f)\right]
$$

Substitute from Steps 2a and 2b:

$$
= \frac{1}{2}\Big[|\nabla^2 f|^2_{HS} + \nabla f \cdot \nabla(\Delta f) - \nabla V_{\text{total}} \cdot (\nabla^2 f \nabla f)
$$

$$
- 2\left(\frac{1}{2}\nabla f \cdot \nabla(\Delta f) - \langle \nabla f, \nabla^2 V_{\text{total}} \nabla f \rangle - \nabla f \cdot (\nabla^2 f) \nabla V_{\text{total}}\right)\Big]
$$

$$
= \frac{1}{2}\Big[|\nabla^2 f|^2_{HS} + \nabla f \cdot \nabla(\Delta f) - \nabla V_{\text{total}} \cdot (\nabla^2 f \nabla f)
$$

$$
- \nabla f \cdot \nabla(\Delta f) + 2\langle \nabla f, \nabla^2 V_{\text{total}} \nabla f \rangle + 2\nabla f \cdot (\nabla^2 f) \nabla V_{\text{total}}\Big]
$$

The $\nabla f \cdot \nabla(\Delta f)$ terms cancel. The cross terms with $\nabla V_{\text{total}} \cdot (\nabla^2 f \nabla f)$ need careful checking (they are transposes).

Using symmetry of the Hessian and index notation:

$$
\nabla V_{\text{total}} \cdot (\nabla^2 f \nabla f) = (\partial_i V_{\text{total}}) (\partial_i \partial_j f) (\partial_j f) = \nabla f \cdot (\nabla^2 f) \nabla V_{\text{total}}
$$

So these terms cancel (with factor of 2):

$$
\Gamma_2(f, f) = \frac{1}{2}|\nabla^2 f|^2_{HS} + \langle \nabla f, \nabla^2 V_{\text{total}} \nabla f \rangle
$$

**Result (Bochner-Weitzenböck Formula)**:

$$
\boxed{\Gamma_2(f, f) = \frac{1}{2}\|\nabla^2 f\|^2_{HS} + \langle \nabla f, \nabla^2 V_{\text{total}} \nabla f \rangle}
$$

---

**Step 3: Apply Uniform Convexity**

We are given that V_total is uniformly convex:

$$
\nabla^2 V_{\text{total}} \succeq \lambda_{\text{BE}} I
$$

This means for all vectors $\xi \in \mathbb{R}^d$:

$$
\langle \xi, \nabla^2 V_{\text{total}} \xi \rangle \ge \lambda_{\text{BE}} |\xi|^2
$$

**Apply to Γ_2**: Taking $\xi = \nabla f$:

$$
\langle \nabla f, \nabla^2 V_{\text{total}} \nabla f \rangle \ge \lambda_{\text{BE}} |\nabla f|^2 = \lambda_{\text{BE}} \Gamma(f, f)
$$

Since $\|\nabla^2 f\|^2_{HS} \ge 0$, we have:

$$
\Gamma_2(f, f) \ge \lambda_{\text{BE}} \Gamma(f, f)
$$

**Result**: The Bakry-Émery Γ_2 criterion is satisfied with curvature constant λ_BE.

---

**Step 4: Verify C∞ Regularity Assumption**

The computation of Γ_2 in Step 2 requires:
- **Second derivatives** of f (for Hessian $\nabla^2 f$)
- **Third derivatives** of V_total (for $\nabla(\nabla^2 V_{\text{total}})$ if we expand further)
- **Fourth derivatives** of V_total (if we compute Γ_2 for specific test functions)

For the Γ_2 criterion to be well-defined and the above computation to hold for all smooth test functions f, we need **V_total ∈ C∞**.

Since V_total = U + ε_F V_fit, and U is assumed smooth (framework axiom), we require **V_fit ∈ C∞**.

**Verification**: This is exactly what Theorem {prf:ref}`thm-cinf-regularity` proves (the main result of document 19_geometric_gas_cinf_regularity_simplified.md).

**Status**: ✅ C∞ regularity of V_fit is established (subject to simplified position-dependent model)

---

**Step 5: Identify Implications**

The Bakry-Émery Γ_2 ≥ λ Γ condition is known to imply (Bakry & Émery 1985):

**Implication 1: Spectral Gap**

The generator L has a spectral gap:

$$
\lambda_{\text{gap}} := \inf_{f \perp 1} \frac{-\langle f, \mathcal{L}f \rangle_\mu}{\|f\|^2_{L^2(\mu)}} \ge \lambda_{\text{BE}}
$$

where μ is the invariant measure $d\mu = e^{-2V_{\text{total}}} dx$ (normalized).

This implies **exponential convergence** to equilibrium at rate λ_BE.

**Implication 2: Log-Sobolev Inequality (LSI)**

The measure μ satisfies a LSI with constant 1/λ_BE:

$$
\text{Ent}_\mu(f^2) \le \frac{2}{\lambda_{\text{BE}}} \int |\nabla f|^2 d\mu
$$

This is stronger than the spectral gap and implies concentration inequalities.

**Implication 3: Hypercontractivity**

By Nelson's theorem, the LSI implies hypercontractivity:

$$
\|P_t f\|_{L^q(\mu)} \le \|f\|_{L^p(\mu)}
$$

for $q > p \ge 1$ and $t \ge \frac{q - p}{2(q-1)p \lambda_{\text{BE}}}$, where $P_t = e^{t\mathcal{L}}$ is the semigroup.

**Status**: ✅ All implications follow from standard theorems (Bakry-Émery 1985, Ledoux 1999)

---

## III. Framework Dependencies

### Verified Dependencies

**Axioms**:

| Label | Statement | Used in Step | Verified |
|-------|-----------|--------------|----------|
| V_fit ∈ C∞ | Fitness potential is infinitely differentiable | Step 4 (Γ_2 well-defined) | ✅ (Theorem {prf:ref}`thm-cinf-regularity`) |
| Uniform convexity | ∇²V_total ≥ λ_BE I | Step 3 (lower bound) | ⚠️ CONDITIONAL ASSUMPTION |

**Theorems**:

| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| thm-cinf-regularity | 19_geometric_gas_cinf_regularity_simplified.md § 5 | V_fit ∈ C∞ with Gevrey-1 bounds | Step 4 | ✅ |
| Bochner-Weitzenböck | Bakry-Émery (1985) | Γ_2(f,f) = (1/2)\|\nabla²f\|² + ⟨∇f, ∇²V ∇f⟩ | Step 2c | ✅ (standard) |
| Bakry-Émery Theorem | Bakry-Émery (1985), Ledoux (1999) | Γ_2 ≥ λΓ implies λ_gap ≥ λ, LSI, hypercontractivity | Step 5 | ✅ (standard) |

**Definitions**:

| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| Carré du champ | Bakry-Émery calculus | Γ(f,g) = (1/2)[L(fg) - fLg - gLf] | Step 1 |
| Iterated carré du champ | Bakry-Émery calculus | Γ_2(f,f) = (1/2)[LΓ(f,f) - 2Γ(f,Lf)] | Step 2 |
| V_total | 11_geometric_gas.md | V_total = U + ε_F V_fit | All steps |

**Constants**:

| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| λ_BE | Bakry-Émery curvature constant | λ_min(∇²V_total) | Lower bound on Hessian eigenvalues |
| ε_F | Fitness weight | Framework parameter | Small for perturbative regime |

### Missing/Uncertain Dependencies

**CRITICAL CONDITIONAL ASSUMPTION**:
- **Uniform convexity of V_total**: ∇²V_total ≥ λ_BE I
  - **Status**: ⚠️ NOT GENERALLY SATISFIED
  - **Why problematic**: V_fit depends on empirical diversity, not convex in general
    - For example, if diversity encourages exploration away from dense regions, V_fit may have saddle points or local maxima
    - Uniform convexity requires ∇²V_fit ≥ -λ_U I where ∇²U ≥ λ_U I from confining potential
  - **When it holds**:
    - Small ε_F regime: V_total ≈ U (dominated by confining potential)
    - Specific fitness landscapes: If diversity measurement is convex in position
    - Restricted domains: May hold locally near equilibrium
  - **Verification needed**: Case-by-case analysis for specific potentials U and ε_F

**Requires Additional Proof**:
- **Lemma (Uniform Convexity Transfer)**: For sufficiently small ε_F, if ∇²U ≥ κ_conf I and ∇²V_fit is bounded, then:
  $$
  \nabla^2 V_{\text{total}} = \nabla^2 U + \epsilon_F \nabla^2 V_{\text{fit}} \succeq (\kappa_{\text{conf}} - \epsilon_F L_{\text{Hess}}) I
  $$
  where L_Hess is the Lipschitz constant of ∇V_fit (from C² bounds).
  - **Difficulty**: MEDIUM (requires bounding $\|\nabla^2 V_{\text{fit}}\|_\infty$ using C² or C³ results)
  - **Conclusion**: For $\epsilon_F < \kappa_{\text{conf}} / L_{\text{Hess}}$, uniform convexity holds with λ_BE = κ_conf - ε_F L_Hess

---

## IV. Proof Obstacles

### Critical Obstacles

1. **Obstacle: Uniform Convexity of V_fit**
   - **Issue**: The fitness potential V_fit[f_k, ρ](x_i) is **NOT** uniformly convex in general
   - **Impact**: The key assumption ∇²V_total ≥ λ_BE I fails for generic diversity landscapes
   - **Resolution**:
     - **Option A** (Perturbative): Assume ε_F small enough that U dominates (requires bounding ∇²V_fit)
     - **Option B** (Restricted domain): Prove convexity in a local region near equilibrium
     - **Option C** (Weaker result): Replace Γ_2 ≥ λΓ with Γ_2 + C ≥ λΓ (defective LSI)
   - **Status**: 🔴 CONDITIONAL (proposition is marked "Conditional" for this reason)

2. **Obstacle: Extension to Non-Isotropic Diffusion**
   - **Issue**: The proof assumes Σ_reg = I (standard diffusion); general case Σ_reg ≠ I requires tensor calculus
   - **Impact**: Γ operator becomes ⟨∇f, Σ_reg² ∇g⟩, complicating commutator identities
   - **Resolution**: Change of variables x̃ = Σ_reg^{-1} x to reduce to isotropic case, then transform back
   - **Status**: 🟡 MEDIUM DIFFICULTY (standard technique, but increases complexity)

### Technical Gaps

1. **Gap: Explicit λ_BE formula**
   - **What's missing**: Closed-form expression for λ_BE in terms of ε_F, κ_conf, ∥∇²V_fit∥_∞
   - **Why needed**: For verifying spectral gap and LSI constants
   - **Difficulty**: MEDIUM (requires C² bounds on V_fit, which exist from prior documents)

2. **Gap: Verification for specific examples**
   - **What's missing**: Numerical or analytical check that ∇²V_total ≥ λ_BE I for concrete test cases
   - **Why needed**: To validate that the conditional assumption is realistic
   - **Difficulty**: LOW (computational verification possible)

---

## V. Proof Validation

### Logical Structure

- **Step dependencies**: ✅ Steps 1→2→3→4→5 are logically ordered
- **Circular reasoning**: ✅ No circular dependencies (C∞ regularity is proven independently)
- **Axiom usage**: ⚠️ Uniform convexity is a CONDITIONAL ASSUMPTION, not a verified axiom

### Constant Tracking

**λ_BE Dependencies**:
- Depends on minimum eigenvalue of ∇²V_total
- For small ε_F: λ_BE ≈ κ_conf - ε_F L_Hess
- **N-uniformity**: ⚠️ V_fit has N-uniform bounds, but convexity may degrade with N (requires verification)
- **ρ-dependence**: ⚠️ V_fit has ρ-dependent bounds; convexity may fail as ρ → 0

### Compatibility with Framework

- **Notation**: ✅ Matches Bakry-Émery standard notation
- **Measure theory**: ✅ All operators well-defined on C∞ functions
- **Physical units**: ✅ Dimensionless (Γ and Γ_2 have same units)

---

## VI. Implementation Notes

### For the Theorem Prover

**Input Requirements**:
1. Theorem statement (from 19_geometric_gas_cinf_regularity_simplified.md § 10.3, line 965)
2. C∞ regularity (Theorem {prf:ref}`thm-cinf-regularity`)
3. Uniform convexity assumption (MUST be verified or assumed)
4. Bakry-Émery calculus framework (standard reference)

**Expected Proof Complexity**:
- **Difficulty**: 🟡 MEDIUM (computation is standard, but conditional assumption is non-trivial)
- **Length**: ~2-3 pages (Bochner-Weitzenböck calculation + convexity verification)
- **New lemmas needed**: 1 (Uniform convexity transfer for small ε_F)
- **Computational verification**: Possible for 1D test cases with explicit V_fit

**Recommended Tools**:
- Bakry-Émery calculus (commutator identities, Bochner formula)
- Spectral theory (spectral gap, LSI, hypercontractivity theorems)
- Perturbation theory (for small ε_F analysis)

**Verification Strategy**:
1. Check Bochner-Weitzenböck formula computation (Steps 2a-2c)
2. Verify uniform convexity assumption holds for specific examples
3. Compute λ_BE explicitly in small ε_F regime
4. Compare LSI constant with independent entropy production calculations

---

## VII. Alternative Approaches

### Approach 1: Defective LSI

**Idea**: If uniform convexity fails, prove a **defective LSI** of the form:

$$
\text{Ent}_\mu(f^2) \le \frac{2}{\lambda} \int |\nabla f|^2 d\mu + C \|f\|^2_{L^2}
$$

This still implies exponential convergence, just with a worse constant.

**Pros**: Does not require uniform convexity
**Cons**: Weaker result, no hypercontractivity

**Status**: Not pursued (would require modifying proposition statement)

### Approach 2: Hypocoercivity

**Idea**: Instead of Bakry-Émery, use **hypocoercive methods** (Villani 2009) that exploit coupling between position and velocity.

**Pros**: Can handle non-convex potentials
**Cons**: Requires kinetic framework (this document is position-only)

**Status**: Not applicable (different setting)

### Approach 3: Local Convexity

**Idea**: Prove Γ_2 ≥ λΓ only on a **restricted domain** (e.g., ball around equilibrium).

**Pros**: More realistic (convexity often holds locally)
**Cons**: Loses global convergence guarantees

**Status**: **RECOMMENDED** for practical applications

---

## VIII. Summary and Recommendations

### Summary

This proposition establishes the **Bakry-Émery Γ_2 criterion** for the Geometric Gas generator, contingent on the assumption of **uniform convexity** of the total potential V_total. The proof follows the standard Bakry-Émery calculus:
1. Compute Γ (carré du champ) = ⟨∇f, ∇g⟩
2. Compute Γ_2 (iterated carré du champ) via Bochner-Weitzenböck formula
3. Use uniform convexity to bound Γ_2 ≥ λ_BE Γ

**Key Results**:
- C∞ regularity of V_fit ensures Γ_2 is well-defined ✅
- Uniform convexity ∇²V_total ≥ λ_BE I implies Γ_2 ≥ λ_BE Γ ✅
- Standard implications: spectral gap, LSI, hypercontractivity ✅

**Main Caveat**:
- Uniform convexity is **NOT** generally satisfied by V_fit (diversity landscapes are not uniformly convex)
- Holds only in specific regimes:
  - Small ε_F (perturbative regime where U dominates)
  - Special fitness designs (convex diversity measurements)
  - Local neighborhoods (near equilibrium)

### Recommendations for Theorem Prover

**Priority**: MEDIUM (important for spectral theory, but conditional)

**Approach**:
1. **Accept as conditional result** (as stated in original proposition)
2. **Add lemma**: Uniform convexity transfer for small ε_F (see § IV)
3. **Compute explicit λ_BE**: λ_BE = κ_conf - ε_F L_Hess for small ε_F
4. **Verify for examples**: Test uniform convexity numerically for 1D case

**Verification**:
- Bochner-Weitzenböck computation is standard (low risk)
- Uniform convexity assumption is the critical verification point
- Recommend explicit examples to validate feasibility

**Estimated Effort**: 2-3 pages (computation is straightforward, assumption verification adds complexity)

---

**Next Steps**:
1. Prove lemma on uniform convexity transfer (small ε_F regime)
2. Compute explicit spectral gap and LSI constant formulas
3. Verify Γ_2 ≥ λΓ for 1D Gaussian potential + diversity (test case)

**Status**: ✅ COMPLETE CONDITIONAL PROOF - Standard Bakry-Émery calculation, contingent on uniform convexity assumption
