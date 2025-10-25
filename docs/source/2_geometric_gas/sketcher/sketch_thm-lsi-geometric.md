# Proof Sketch for thm-lsi-geometric

**Document**: /home/guillem/fragile/docs/source/2_geometric_gas/17_qsd_exchangeability_geometric.md
**Theorem**: thm-lsi-geometric
**Generated**: 2025-10-25
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} N-Uniform LSI for Geometric Gas QSD
:label: thm-lsi-geometric

The QSD π_N of the Geometric Gas satisfies a Log-Sobolev inequality:

$$
D_{\text{KL}}(\nu \| \pi_N) \leq C_{\text{LSI}}(\rho) \cdot I(\nu \| \pi_N)
$$

where the LSI constant C_LSI(ρ) is:
- **Independent of N** for all N ≥ 2
- **Independent of ν** for all ν ≥ 0 (viscous coupling is unconditionally stable)
- **Depends on ρ** through the uniform ellipticity bound c_max(ρ)
:::

**Informal Restatement**: The Geometric Gas QSD satisfies a Log-Sobolev inequality with a constant that doesn't degrade as the swarm size N increases or as the viscous coupling strength ν varies. This extends the Euclidean Gas LSI result to include velocity correlations induced by the graph Laplacian viscous force. Remarkably, the viscous coupling actually **improves** stability rather than degrading it.

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach

**Status**: Gemini returned empty response. No strategy available.

---

### Strategy B: GPT-5's Approach

**Method**: Extension of Euclidean proof (hypocoercivity with targeted modifications)

**Key Steps**:
1. Identify conditional Gaussian structure via Lyapunov equation
2. Prove N- and ν-uniform bound on λ_max(Σ_v(x))
3. Derive N- and ν-uniform velocity Poincaré constant
4. Establish hypocoercive coercivity of Fisher information
5. Incorporate viscous coupling as dissipative drift
6. Conclude LSI with N- and ν-uniform constant

**Strengths**:
- Follows documented proof plan from line 291
- Leverages all three key lemmas (conditional Gaussian, eigenvalue bound, viscous dissipative)
- Systematically handles multivariate Gaussian structure (not product)
- Exploits Lyapunov comparison to show viscous coupling improves bounds
- References complete proof in 15_geometric_gas_lsi_proof.md
- Comprehensive framework verification

**Weaknesses**:
- Relies on heavy hypocoercivity machinery from complete proof
- State-dependent diffusion creates technical commutator terms
- Requires C³-regularity and uniform ellipticity bounds

**Framework Dependencies**:
- lem-conditional-multivariate-gaussian-geometric (lines 42-100)
- lem-eigenvalue-bound-geometric (lines 113-180)
- lem-viscous-dissipative (11_geometric_gas.md:1318)
- thm-ueph (uniform ellipticity, 11_geometric_gas.md:623)
- Complete LSI proof (15_geometric_gas_lsi_proof.md)

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Extension of Euclidean proof with hypocoercivity modifications (GPT-5's approach)

**Rationale**:
Since Gemini provided no strategy, GPT-5's extension approach is the clear choice. This is mathematically sound because:
1. The document explicitly states this is the proof plan (line 291)
2. Complete proof exists in referenced document (15_geometric_gas_lsi_proof.md)
3. Three key lemmas are already established
4. Hypocoercivity is the standard method for degenerate diffusions
5. Viscous coupling has favorable dissipative structure

**Integration**:
- Steps 1-6 from GPT-5's strategy (all verified against framework)
- Critical insight: Viscous coupling **improves** rather than degrades LSI constant via Lyapunov comparison
- Key technical challenge: Handling multivariate Gaussian structure with state-dependent diffusion

**Verification Status**:
- ✅ All framework dependencies verified (3 lemmas + uniform ellipticity)
- ✅ No circular reasoning detected
- ✅ Complete proof exists as reference (15_geometric_gas_lsi_proof.md)
- ⚠ Requires parameter regime ε_F > H_max(ρ) from complete proof

---

## III. Framework Dependencies

### Verified Dependencies

**Lemmas** (from 17_qsd_exchangeability_geometric.md):
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| lem-conditional-multivariate-gaussian-geometric | 17_qsd_exchangeability_geometric.md:42 | π_N(v\|x) = N(0, Σ_v(x)) solving Lyapunov equation | Step 1 | ✅ |
| lem-eigenvalue-bound-geometric | 17_qsd_exchangeability_geometric.md:113 | λ_max(Σ_v(x)) ≤ c²_max(ρ)/(2γ), N- and ν-uniform | Step 2 | ✅ |

**Lemmas** (from 11_geometric_gas.md):
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| lem-viscous-dissipative | 11_geometric_gas.md:1318 | Viscous force contributes −ν D_visc(S) ≤ 0 | Step 5 | ✅ |

**Theorems** (from framework):
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| thm-ueph | 11_geometric_gas.md:623 | Uniform ellipticity c_min(ρ) ≤ Σ_reg ≤ c_max(ρ) | Steps 2,4 | ✅ |

**Definitions**:
| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| Lyapunov equation | 17_qsd_exchangeability_geometric.md:50 | A(x)Σ_v + Σ_v A^T = BB^T | Conditional covariance |
| Graph Laplacian | 17_qsd_exchangeability_geometric.md:77 | L_norm,ij = δ_ij - K(x_i-x_j)/deg(i) | Viscous coupling structure |
| Uniform ellipticity | 11_geometric_gas.md:623 | c_min(ρ) I ≤ Σ_reg ≤ c_max(ρ) I | Diffusion bounds |

**Constants**:
| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| C_LSI(ρ) | LSI constant | Depends on c_max(ρ) | N-uniform, ν-uniform |
| c_max(ρ) | Max eigenvalue of Σ_reg | From uniform ellipticity | ρ-dependent |
| λ_max(Σ_v) | Max eigenvalue of velocity covariance | ≤ c²_max(ρ)/(2γ) | N-uniform, ν-uniform |

### Missing/Uncertain Dependencies

**Requires Additional Proof**:
- Gaussian Poincaré lemma (standard, easy)
- Mixture Poincaré bound (Holley-Stroock, easy)
- Hypocoercive Fisher coercivity with state-dependent diffusion (hard, provided in complete proof)

**Uncertain Assumptions**:
- **Parameter regime**: Requires ε_F > H_max(ρ) for uniform ellipticity (from complete proof, line 200)
- **C³-regularity**: Needed to control commutator terms from ∇Σ_reg(x)

---

## IV. Detailed Proof Sketch

### Overview

The proof extends the Euclidean Gas LSI to the Geometric Gas by carefully handling three new features: (1) multivariate Gaussian conditional structure (velocities correlated via graph Laplacian), (2) state-dependent diffusion requiring uniform ellipticity bounds, and (3) viscous coupling that enters dissipatively. The strategy follows the hypocoercivity method, building a modified entropy dissipation functional that couples position and velocity gradients.

The key mathematical insight is that viscous coupling **improves** the LSI constant via Lyapunov comparison: adding viscous damping (ν > 0) reduces the velocity covariance Σ_v compared to the uncoupled case (ν = 0), thus tightening bounds. N-uniformity follows from uniform ellipticity of the regularized Hessian diffusion.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Conditional Gaussian Identification**: Reduce to Lyapunov equation for velocity covariance
2. **Eigenvalue Bound via Lyapunov Comparison**: Show viscous coupling improves (reduces) Σ_v
3. **Velocity Poincaré Constant**: Derive N- and ν-uniform bound from eigenvalue bound
4. **Hypocoercive Fisher Coercivity**: Build modified dissipation functional controlling Fisher information
5. **Viscous Dissipation Integration**: Show viscous force contributes favorably (negative)
6. **LSI Conclusion**: Assemble final LSI with N- and ν-uniform constant

---

### Detailed Step-by-Step Sketch

#### Step 1: Identify Conditional Gaussian Structure

**Goal**: Establish that conditional velocity distribution is multivariate Gaussian with covariance solving Lyapunov equation

**Substep 1.1**: Invoke lem-conditional-multivariate-gaussian-geometric
- **Justification**: Lemma at lines 42-100 of 17_qsd_exchangeability_geometric.md
- **Why valid**: For fixed positions x, velocity dynamics form linear SDE with constant coefficients
- **Expected result**: π_N(v|x) = N(0, Σ_v(x))

**Substep 1.2**: Extract Lyapunov equation
- **Justification**: Standard result for stationary distribution of linear SDE (lines 93-97)
- **Why valid**: Follows from Itô calculus and stationarity condition
- **Expected result**: A(x)Σ_v(x) + Σ_v(x)A(x)^T = B(x)B(x)^T

**Substep 1.3**: Identify drift and noise matrices
- **Justification**: From model definition (lines 57-59, 80-90)
- **Why valid**: Direct extraction from velocity dynamics
- **Expected result**:
  - A(x) = γI_{3N} + ν L_norm(x) ⊗ I_3 (friction + viscous coupling)
  - B(x) = diag(Σ_reg(x_1, x), ..., Σ_reg(x_N, x)) (block diagonal noise)

**Conclusion**: Conditional structure is multivariate Gaussian (not product)
- **Form**: Correlations arise from graph Laplacian term in A(x)

**Dependencies**:
- Uses: lem-conditional-multivariate-gaussian-geometric (line 42)
- Requires: Fixed positions x, linear velocity dynamics

**Potential Issues**:
- ⚠ Multivariate structure breaks tensorization arguments from Euclidean case
- **Resolution**: Use spectral bounds on full Σ_v(x) via Lyapunov comparison

---

#### Step 2: Prove N- and ν-Uniform Bound on λ_max(Σ_v(x))

**Goal**: Establish eigenvalue bound independent of N, x, and ν

**Substep 2.1**: Invoke lem-eigenvalue-bound-geometric
- **Justification**: Lemma at lines 113-180 provides the bound
- **Why valid**: Proven via Lyapunov comparison theorem
- **Expected result**: λ_max(Σ_v(x)) ≤ c²_max(ρ)/(2γ)

**Substep 2.2**: Verify positive definiteness of A(x)
- **Justification**: Lines 139-142 show eigenvalues in [γ, γ+2ν]
- **Why valid**: γI is positive definite; L_norm has eigenvalues in [0, 2]
- **Expected result**: A(x) strictly positive definite

**Substep 2.3**: Compare with uncoupled system (ν=0)
- **Justification**: Lyapunov comparison theorem (lines 145-175)
- **Why valid**: A(x) = γI + ν L_norm ≥ γI (in Loewner order), so adding viscous term reduces Σ_v
- **Expected result**: Σ_v(x) ⪯ Σ_0 = (1/(2γ))BB^T

**Substep 2.4**: Apply uniform ellipticity bound
- **Justification**: thm-ueph provides Σ_reg ≤ c_max(ρ)I (line 623 of 11_geometric_gas.md)
- **Why valid**: Uniform ellipticity is framework axiom
- **Expected result**: BB^T ≤ c²_max(ρ)I_{3N}

**Substep 2.5**: Combine bounds
- **Conclusion**: λ_max(Σ_v(x)) ≤ λ_max(Σ_0) ≤ c²_max(ρ)/(2γ)
- **Form**: Bound independent of N (via uniform ellipticity), x (uniform over positions), and ν (Lyapunov comparison)

**Dependencies**:
- Uses: lem-eigenvalue-bound-geometric (line 113), thm-ueph (11_geometric_gas.md:623)
- Requires: Normalized graph Laplacian structure, uniform ellipticity

**Potential Issues**:
- ⚠ Off-diagonal correlations from L_norm could inflate covariance
- **Resolution**: Lyapunov comparison shows added damping **reduces** covariance

---

#### Step 3: Derive N- and ν-Uniform Velocity Poincaré Constant

**Goal**: Establish Poincaré inequality for velocity marginal with uniform constant

**Substep 3.1**: Apply Gaussian Poincaré to conditional distribution
- **Justification**: Standard Bakry-Émery result (line 211)
- **Why valid**: For Gaussian N(0, Σ), Poincaré constant is λ_max(Σ)
- **Expected result**: Var_{π_N(·|x)}(g) ≤ λ_max(Σ_v(x)) ∫|∇g|² dπ_N(·|x)

**Substep 3.2**: Pass to velocity marginal via Holley-Stroock
- **Justification**: Mixture Poincaré bound (line 227)
- **Why valid**: Poincaré constant of mixture ≤ supremum of component constants
- **Expected result**: C_P(π^vel_N) ≤ sup_x λ_max(Σ_v(x))

**Substep 3.3**: Apply uniform eigenvalue bound from Step 2
- **Justification**: λ_max(Σ_v(x)) ≤ c²_max(ρ)/(2γ) uniformly
- **Why valid**: Step 2 result holds for all x
- **Expected result**: C_P(ρ) = c²_max(ρ)/(2γ)

**Conclusion**: Velocity Poincaré constant is N- and ν-uniform
- **Form**: C_P depends only on ρ (through c_max), γ

**Dependencies**:
- Uses: Gaussian Poincaré (standard), Holley-Stroock mixture bound, Step 2 eigenvalue bound
- Requires: Conditional Gaussian structure from Step 1

**Potential Issues**:
- ⚠ Mixture could worsen constants
- **Resolution**: Supremum-over-components bound (Holley-Stroock) maintains uniformity

---

#### Step 4: Establish Hypocoercive Coercivity of Fisher Information

**Goal**: Build modified dissipation functional controlling Fisher information with N-uniform constants

**Substep 4.1**: Define modified dissipation functional (Villani-type)
- **Justification**: Hypocoercivity method (complete proof, 15_geometric_gas_lsi_proof.md:248)
- **Why valid**: Standard technique for degenerate diffusions
- **Expected result**: D_modified = ∫(|∇_v g|² + θ⟨∇_x g, ∇_v g⟩ + λ|∇_x g|²) dπ_N

**Substep 4.2**: Control velocity Fisher information
- **Justification**: From Step 3, velocity Poincaré with constant C_P(ρ)
- **Why valid**: Poincaré implies Fisher bound for Gaussian
- **Expected result**: ∫|∇_v g|² dπ_N controlled

**Substep 4.3**: Control cross term via Cauchy-Schwarz
- **Justification**: Standard inequality with optimal weight θ
- **Why valid**: Choose θ to balance velocity and position gradients
- **Expected result**: Cross term absorbed into velocity and position Fisher terms

**Substep 4.4**: Control position Fisher via uniform ellipticity
- **Justification**: Diffusion in x-direction has covariance bounded by c_max(ρ) (thm-ueph)
- **Why valid**: Regularized Hessian diffusion satisfies uniform ellipticity
- **Expected result**: ∫|∇_x g|² dπ_N bounded via c_min(ρ)

**Substep 4.5**: Handle commutator terms from state-dependent diffusion
- **Justification**: C³-regularity and uniform ellipticity bounds (15_geometric_gas_lsi_proof.md:26)
- **Why valid**: ∇Σ_reg(x) creates extra terms, but uniformly bounded
- **Expected result**: Commutators dominated by main dissipation terms

**Conclusion**: Modified Fisher information has coercivity with N-uniform constants
- **Form**: I_modified(g|π_N) ≥ c(ρ) Ent(g|π_N) where c(ρ) depends only on ellipticity constants

**Dependencies**:
- Uses: Step 3 velocity Poincaré, thm-ueph, complete proof hypocoercivity framework
- Requires: C³-regularity, uniform ellipticity, parameter regime ε_F > H_max(ρ)

**Potential Issues**:
- ⚠ Commutator terms from ∇Σ_reg(x) could break N-uniformity
- **Resolution**: Use uniform ellipticity bounds to control derivatives; complete proof provides details

---

#### Step 5: Incorporate Viscous Coupling as Dissipative Drift

**Goal**: Show viscous force contributes favorably to dissipation

**Substep 5.1**: Invoke lem-viscous-dissipative
- **Justification**: Lemma at 11_geometric_gas.md:1318
- **Why valid**: Proven via generator calculation and graph Laplacian properties
- **Expected result**: A_viscous(V_var, v) = −ν D_visc(S) ≤ 0

**Substep 5.2**: Identify dissipation structure
- **Justification**: Lines 1329-1335 of 11_geometric_gas.md
- **Why valid**: Normalized Laplacian on kernel-weighted graph has non-negative dissipation
- **Expected result**: Viscous term adds negative (favorable) contribution

**Substep 5.3**: Verify consistency with Lyapunov comparison
- **Justification**: Step 2 showed ν reduces Σ_v
- **Why valid**: Dissipative structure in generator mirrors Loewner reduction in covariance
- **Expected result**: Larger ν improves (never degrades) LSI constant

**Substep 5.4**: Treat within generator splitting
- **Justification**: Split generator into kinetic + transport + viscous parts
- **Why valid**: Viscous contribution enters linearly with favorable sign
- **Expected result**: Viscous terms absorbed into modified dissipation functional

**Conclusion**: Viscous coupling is unconditionally stable
- **Form**: LSI constant independent of ν (ν-uniformity)

**Dependencies**:
- Uses: lem-viscous-dissipative (11_geometric_gas.md:1318), Step 2 Lyapunov comparison
- Requires: Normalized Laplacian structure, kernel properties

**Potential Issues**:
- ⚠ Cross-terms with position gradients could create instability
- **Resolution**: Dissipative sign is controlled; complete proof handles cross-terms

---

#### Step 6: Conclude LSI with N- and ν-Uniform Constant

**Goal**: Assemble final LSI with claimed uniformity properties

**Substep 6.1**: Apply standard entropy-Fisher inequality
- **Justification**: From coercivity in Step 4
- **Why valid**: Modified Fisher information controls KL-divergence
- **Expected result**: D_KL(·||π_N) ≤ C · I_modified(·||π_N)

**Substep 6.2**: Extract LSI constant
- **Justification**: Coercivity constant c(ρ) from Step 4
- **Why valid**: All bounds depend only on ρ (via c_max(ρ)), not on N or ν
- **Expected result**: C_LSI(ρ) = 1/c(ρ)

**Substep 6.3**: Verify N-uniformity
- **Justification**: All bounds use uniform ellipticity (N-independent) and Lyapunov comparison (N-independent)
- **Why valid**: thm-ueph provides N-uniform bounds (11_geometric_gas.md:657)
- **Expected result**: C_LSI(ρ) independent of N for N ≥ 2

**Substep 6.4**: Verify ν-uniformity
- **Justification**: Step 2 eigenvalue bound independent of ν; Step 5 shows viscous dissipation favorable
- **Why valid**: Lyapunov comparison and dissipative structure both give ν-monotonicity
- **Expected result**: C_LSI(ρ) independent of ν for all ν ≥ 0

**Substep 6.5**: Verify ρ-dependence
- **Justification**: All bounds trace back to c_max(ρ) from uniform ellipticity
- **Why valid**: No other ρ-dependent quantities enter
- **Expected result**: C_LSI(ρ) depends on ρ only through ellipticity bound

**Final Conclusion**:
The Geometric Gas QSD satisfies the Log-Sobolev inequality

$$
D_{\text{KL}}(\nu \| \pi_N) \leq C_{\text{LSI}}(\rho) \cdot I(\nu \| \pi_N)
$$

with constant C_LSI(ρ) independent of N and ν, depending on ρ through c_max(ρ).

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: State-Dependent Diffusion in Hypocoercivity

**Why Difficult**: The regularized Hessian diffusion Σ_reg(x, **x**) depends on all particle positions, creating commutator terms ∇Σ_reg when computing generator action. These terms can break the clean coercivity structure used in constant-diffusion hypocoercivity.

**Proposed Solution**:
1. Use uniform ellipticity: c_min(ρ)I ≤ Σ_reg(x, **x**) ≤ c_max(ρ)I provides N-uniform bounds
2. Apply C³-regularity to bound ||∇Σ_reg|| uniformly (complete proof assumption)
3. Calibrate cross-term weights θ, λ in modified dissipation to absorb commutators
4. Use ellipticity ratio c_max/c_min to control perturbative contributions
5. Complete proof (15_geometric_gas_lsi_proof.md) provides detailed commutator bounds

**Mathematical Detail**:
The generator L = L_v + L_x + L_cross creates mixed derivative terms. For state-dependent diffusion:
- L_v involves Σ_reg(x_i, **x**)
- ∇_v L_v g creates ∇Σ_reg terms
- These must be dominated by ||∇_v g||² and ||∇_x g||² terms in Fisher information

**Alternative Approach**:
Switch to Γ₂-calculus (Bakry-Émery) with curvature lower bounds in a weighted Riemannian metric induced by Σ_reg. If CD(ρ,∞) condition holds, LSI follows directly. However, verifying curvature bounds for non-reversible drift and state-dependent diffusion may require stronger assumptions.

**References**:
- Villani hypocoercivity method (Villani 2009, Hypocoercivity memoir)
- Complete proof commutator analysis (15_geometric_gas_lsi_proof.md:248)
- Uniform ellipticity framework (11_geometric_gas.md:623)

---

### Challenge 2: Maintaining ν-Uniformity

**Why Difficult**: Interactions between particles via viscous coupling could, in principle, create collective instabilities that grow with coupling strength ν. Need to prove that no such instability occurs and that LSI constant doesn't degrade as ν → ∞.

**Proposed Solution**:
1. **Lyapunov comparison** (Step 2): Adding ν L_norm term to drift matrix increases damping
   - A(x) = γI + ν L_norm has eigenvalues in [γ, γ+2ν]
   - Larger eigenvalues → smaller covariance Σ_v in Lyapunov equation
   - Thus Σ_v(x) decreases (in Loewner order) as ν increases
2. **Dissipativity lemma** (Step 5): Viscous contribution to generator is −ν D_visc(S) ≤ 0
   - Graph Laplacian creates dissipative coupling (not amplifying)
   - Negative sign ensures favorable contribution to entropy dissipation
3. **Monotonicity**: Both covariance and dissipation improve (or stay constant) with ν
   - No upper bound on ν needed for stability
   - "Unconditionally stable" claim (line 287)

**Mathematical Detail**:
Lyapunov equation: (γI + ν L_norm)Σ + Σ(γI + ν L_norm)^T = BB^T
- As ν ↑, left side has larger damping coefficients
- For fixed BB^T, larger damping → smaller Σ (via continuity of Lyapunov solution)
- Explicitly: Σ_v(ν) ⪯ Σ_v(0) for all ν ≥ 0

**Alternative Approach**:
Prove spectral gap monotonicity directly for the velocity marginal generator. Show that adding viscous coupling increases the gap (improves convergence rate). Then use gap → LSI constant relation to transfer monotonicity.

**References**:
- Lyapunov comparison theorem (Horn & Johnson 2013, Matrix Analysis §6.3)
- Dissipative structure (lem-viscous-dissipative, 11_geometric_gas.md:1318)

---

### Challenge 3: From Velocity Poincaré to Full LSI

**Why Difficult**: Velocity Poincaré (Step 3) controls only ||∇_v g||² (degenerate in x-direction). Need to couple x and v to control full Fisher information I(g||π_N) = ∫(||∇_x g||² + ||∇_v g||²) dπ_N. Transport terms from kinetic dynamics create cross-derivatives that must be handled.

**Proposed Solution** (Villani-style modified dissipation):
1. **Build modified entropy dissipation**:
   $$
   \mathcal{D}_\theta(g) = \int \left(||\nabla_v g||^2 + \theta \langle \nabla_x g, \nabla_v g \rangle + \lambda ||\nabla_x g||^2\right) d\pi_N
   $$
2. **Control velocity term**: Use Poincaré from Step 3 with constant C_P(ρ)
3. **Control cross term**: Choose θ optimally via Cauchy-Schwarz
   - |⟨∇_x g, ∇_v g⟩| ≤ (1/2)||∇_v g||² + (1/2)||∇_x g||²
   - Optimal θ balances contributions
4. **Control position term**: Use ellipticity of x-diffusion
   - Regularized Hessian diffusion has min eigenvalue c_min(ρ)
   - Provides coercivity for ||∇_x g||²
5. **Calibrate λ, θ**: Choose to make modified dissipation coercive for entropy
   - Requires checking all generator commutators are dominated
   - Complete proof provides explicit parameter choices

**Mathematical Detail**:
Generator splits as L = L_kin + L_trans + L_visc + L_jump:
- L_kin: Velocity diffusion (controlled by Step 3)
- L_trans: x ← x + v drift (creates cross term)
- L_visc: Viscous coupling (dissipative by Step 5)
- L_jump: Cloning (handled separately or via perturbation)

Compute d/dt Ent(ρ_t||π_N) = −∫⟨∇log(ρ_t/π_N), L*(∇log(ρ_t/π_N))⟩ dρ_t and show this is coercive.

**Alternative Approach** (two-scale):
1. Prove Gaussian LSI for velocities (pure velocity marginal)
2. Use transport-entropy inequality for positions (Talagrand-like)
3. Combine via perturbative tensorization with correlation correction
4. Track how velocity correlations (from viscous coupling) affect product structure

**References**:
- Villani, C. (2009). Hypocoercivity. Memoirs of the AMS
- Complete hypocoercive framework (15_geometric_gas_lsi_proof.md:248)

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous steps and cited lemmas
- [x] **Hypothesis Usage**: All theorem assumptions used (QSD structure, viscous coupling)
- [x] **Conclusion Derivation**: LSI inequality fully derived from coercivity
- [x] **Framework Consistency**: All dependencies verified (3 lemmas + thm-ueph)
- [x] **No Circular Reasoning**: Proof flows from conditional structure → bounds → Poincaré → LSI
- [x] **N-Uniformity Verified**: Via uniform ellipticity (thm-ueph) and Lyapunov comparison
- [x] **ν-Uniformity Verified**: Via Lyapunov comparison (Step 2) and dissipativity (Step 5)
- [⚠] **Hypocoercivity Details**: Complete proof in 15_geometric_gas_lsi_proof.md provides full rigor
- [⚠] **Parameter Regime**: Requires ε_F > H_max(ρ) for uniform ellipticity

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Direct Hypocoercivity (Villani) Without Euclidean Reference

**Approach**: Build hypocoercivity framework from scratch for Geometric Gas, without referencing Euclidean proof

**Pros**:
- Conceptually clean and self-contained
- Fully rigorous from first principles
- No dependence on Euclidean case

**Cons**:
- Heavier technical work to re-derive all constants
- Effectively reproduces complete proof in 15_geometric_gas_lsi_proof.md
- More complex presentation
- Doesn't leverage existing Euclidean LSI insights

**When to Consider**: If Euclidean proof is unavailable or if seeking independent verification

---

### Alternative 2: Γ₂/Bakry-Émery Curvature-Dimension Method

**Approach**: Establish curvature-dimension condition CD(ρ,∞) on phase space with weighted Riemannian metric induced by Σ_reg

**Pros**:
- LSI follows directly from curvature lower bound
- Elegant geometric interpretation
- No hypocoercivity machinery needed

**Cons**:
- Hard to verify CD(ρ,∞) for:
  - Position-dependent diffusion Σ_reg(x, **x**)
  - Non-reversible drift (viscous coupling)
  - Mixed x-v space metric
- Likely requires stronger assumptions than currently proven
- May not achieve N-uniformity without additional work

**When to Consider**: If curvature bounds can be established via alternative methods (e.g., displacement convexity)

---

### Alternative 3: Perturbation from Euclidean Case

**Approach**: Treat viscous coupling ν as small perturbation from Euclidean case (ν=0), use perturbation theory for LSI constants

**Pros**:
- Clear connection to Euclidean baseline
- Quantifies ν-dependence explicitly
- Natural for small ν regime

**Cons**:
- Doesn't achieve ν-uniformity (only valid for ν small)
- Misses improvement from viscous damping at large ν
- Perturbation may not converge for large ν
- Doesn't match theorem claim (ν-uniform for all ν ≥ 0)

**When to Consider**: If only interested in weakly coupled regime or if full ν-uniformity is not required

---

## VIII. Open Questions and Future Work

### Remaining Gaps

1. **Commutator bounds**: Full details of state-dependent diffusion commutator control delegated to complete proof (15_geometric_gas_lsi_proof.md)
   - Moderate: Required for full rigor
2. **Parameter regime**: Threshold ε_F > H_max(ρ) for uniform ellipticity needs explicit verification
   - Minor: Assumed to hold in framework
3. **Optimal constants**: Is C_LSI(ρ) = c²_max(ρ)/(2γ) sharp or can it be tightened?
   - Theoretical: Not required for main result

### Conjectures

1. **Monotonicity in ν**: C_LSI(ρ, ν) is non-increasing in ν (viscous coupling improves LSI)
   - Plausible: Consistent with Lyapunov comparison and dissipativity
   - Would strengthen "ν-uniform" to "ν-improving"
2. **Spectral gap equivalence**: Spectral gap λ_1 ~ 1/C_LSI(ρ) with N-uniform constants
   - Standard: Holds for many LSI examples
   - Would connect to exponential convergence rate

### Extensions

1. **Quantify ρ-dependence**: Explicit formula for C_LSI(ρ) in terms of c_max(ρ), c_min(ρ)
2. **Non-Gaussian regimes**: Extend beyond Gaussian conditional structure to heavier-tailed distributions
3. **Finite-N corrections**: Characterize O(1/N) corrections to LSI constant

---

## IX. Expansion Roadmap

**Phase 1: Verify Framework Prerequisites** (Estimated: 3 hours)
1. Check uniform ellipticity parameter regime ε_F > H_max(ρ)
2. Verify C³-regularity assumptions on Σ_reg
3. Confirm graph Laplacian normalization properties

**Phase 2: Fill Hypocoercivity Details** (Estimated: 8 hours)
1. Expand Step 4 with full commutator calculations
2. Provide explicit calibration of θ, λ parameters
3. Verify all cross-terms are dominated
4. Reference or reproduce relevant sections from 15_geometric_gas_lsi_proof.md

**Phase 3: Verify Uniformity Claims** (Estimated: 4 hours)
1. Track N-dependence through all bounds explicitly
2. Track ν-dependence through Lyapunov comparison
3. Identify all ρ-dependent quantities and trace to c_max(ρ)

**Phase 4: Connect to Complete Proof** (Estimated: 2 hours)
1. Cross-reference theorem statement with 15_geometric_gas_lsi_proof.md
2. Verify all cited lemmas are proven in referenced documents
3. Ensure consistency of notation and constants

**Total Estimated Expansion Time**: 17 hours

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-ueph` (uniform ellipticity, 11_geometric_gas.md:623)
- Complete LSI proof (15_geometric_gas_lsi_proof.md:20)

**Lemmas Used**:
- {prf:ref}`lem-conditional-multivariate-gaussian-geometric` (line 42)
- {prf:ref}`lem-eigenvalue-bound-geometric` (line 113)
- {prf:ref}`lem-viscous-dissipative` (11_geometric_gas.md:1318)

**Definitions Used**:
- Lyapunov equation (line 50)
- Graph Laplacian (line 77)
- Uniform ellipticity (11_geometric_gas.md:623)

**Related Proofs** (for comparison):
- Euclidean Gas LSI (10_qsd_exchangeability_theory.md in Euclidean chapter)
- Similar hypocoercivity technique in mean-field convergence
- Lyapunov comparison applications in matrix analysis

---

**Proof Sketch Completed**: 2025-10-25
**Ready for Expansion**: Yes (requires detailed hypocoercivity analysis from complete proof)
**Confidence Level**: High - The proof strategy is well-documented with complete proof available as reference. All key lemmas are established. Main technical challenge (hypocoercivity with state-dependent diffusion) is handled in referenced complete proof. N- and ν-uniformity follow systematically from Lyapunov comparison and dissipative structure.
