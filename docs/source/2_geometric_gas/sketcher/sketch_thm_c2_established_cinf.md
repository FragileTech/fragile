# Proof Sketch: C² Regularity and k-Uniform Hessian Bound

**Theorem Label**: `thm-c2-established-cinf` (referenced from 19_geometric_gas_cinf_regularity_simplified.md)
**Source Theorem**: `thm-c2-regularity` from 11_geometric_gas.md § A.4
**Generated**: 2025-10-25
**Agent**: Proof Sketcher (Autonomous Mathematics Pipeline)
**Review Strategy**: Single-strategist (GPT-5/Codex only) due to Gemini MCP issues

---

## Theorem Statement

:::{prf:theorem} C² Regularity and k-Uniform Hessian Bound
:label: thm-c2-regularity

The ρ-localized fitness potential $V_{\text{fit}}[f_k, \rho](x_i)$ is C² in $x_i$ with Hessian satisfying:

$$
\|\nabla^2_{x_i} V_{\text{fit}}[f_k, \rho](x_i)\| \le H_{\max}(\rho)
$$

where $H_{\max}(\rho)$ is a **k-uniform** (and thus **N-uniform**) ρ-dependent constant given by:

$$
H_{\max}(\rho) = L_{g''_A} \|\nabla Z_\rho\|^2_{\max}(\rho) + L_{g_A} \|\nabla^2 Z_\rho\|_{\max}(\rho)
$$

with:
- $\|\nabla Z_\rho\|_{\max}(\rho) = F_{\text{adapt,max}}(\rho) / L_{g_A}$ from Theorem {prf:ref}`thm-c1-regularity` (k-uniform)
- $\|\nabla^2 Z_\rho\|_{\max}(\rho)$ is the **k-uniform** bound on the Hessian of the Z-score (derived below)

**k-Uniform Explicit Bound**: For the Gaussian kernel with bounded measurements, using the **telescoping property** of normalized weights over alive walkers, $H_{\max}(\rho) = O(1/\rho^2)$ and is **independent of k** (and thus of N).
:::

**Full Hypotheses**:
1. State space $\mathcal{X} \subset \mathbb{R}^d$ compact with $C^\infty$ boundary
2. Measurement function $d: \mathcal{X} \to \mathbb{R}$ satisfies $d \in C^\infty(\mathcal{X})$ with $\|d\|_\infty, \|\nabla d\|_\infty, \|\nabla^2 d\|_\infty < \infty$
3. Gaussian localization kernel $K_\rho(r) = \exp(-r^2/(2\rho^2))$ is real analytic
4. Rescale function $g_A: \mathbb{R} \to [0, A]$ satisfies $g_A \in C^\infty$ with bounded derivatives
5. Regularized standard deviation $\sigma'_{\text{reg}}: \mathbb{R}_{\ge 0} \to [\epsilon_\sigma, \infty)$ satisfies $\sigma'_{\text{reg}} \in C^\infty$ with $\sigma'_{\text{reg}} \ge \epsilon_\sigma > 0$
6. Normalized localization weights satisfy $\sum_{j \in A_k} w_{ij}(\rho) = 1$ identically

---

## Proof Strategy Outline

### High-Level Approach

**Method**: Direct chain rule iteration with quotient rule expansion and telescoping mechanism

**Core Insight**: The fitness potential $V_{\text{fit}} = g_A \circ Z_\rho$ where $Z_\rho = (d(x_i) - \mu_\rho) / \sigma'_\rho$ is a quotient of localized moments. The second derivative is computed via:

1. **Outer layer**: Chain rule on $V_{\text{fit}} = g_A(Z_\rho)$ yields $\nabla^2 V_{\text{fit}} = g''_A(Z_\rho) (\nabla Z_\rho) \otimes (\nabla Z_\rho) + g'_A(Z_\rho) \nabla^2 Z_\rho$

2. **Middle layer**: Quotient rule on $Z_\rho = (d - \mu_\rho) / \sigma'_\rho$ produces four term types:
   - Primary Hessian: $(1/\sigma'_\rho)(\nabla^2 d - \nabla^2 \mu_\rho)$
   - Cross products: $(1/(\sigma'_\rho)^2)[(\nabla d - \nabla \mu_\rho) \otimes \nabla \sigma'_\rho + \text{symm}]$
   - Correction: $((d - \mu_\rho)/(\sigma'_\rho)^2) \nabla^2 \sigma'_\rho$
   - Triple-denominator: $(2(d - \mu_\rho)/(\sigma'_\rho)^3) \nabla \sigma'_\rho \otimes \nabla \sigma'_\rho$

3. **Inner layer**: Bounds on $\nabla^2 \mu_\rho$, $\nabla^2 \sigma^2_\rho$ using **telescoping identities**:
   $$\sum_{j \in A_k} \nabla^m w_{ij}(\rho) = 0 \quad \text{for all } m \ge 1$$

   This eliminates k-dependence by converting extensive sums $\sum_j \nabla^2 w_{ij} \cdot d(x_j)$ into centered sums $\sum_j \nabla^2 w_{ij} \cdot (d(x_j) - \mu_\rho)$ which are bounded independently of k.

### Key Technical Steps

**Step 1**: Apply chain rule to $V_{\text{fit}} = g_A(Z_\rho)$
- **Result**: $\nabla^2 V_{\text{fit}} = g''_A (\nabla Z_\rho)^{\otimes 2} + g'_A \nabla^2 Z_\rho$
- **Bound**: $\|\nabla^2 V_{\text{fit}}\| \le L_{g''_A} \|\nabla Z_\rho\|^2 + L_{g_A} \|\nabla^2 Z_\rho\|$

**Step 2**: Apply quotient rule to $Z_\rho = (d - \mu_\rho)/\sigma'_\rho$
- **Result**: Four-term expansion (see middle layer above)
- **Challenge**: Each term has different ρ-scaling, must all aggregate to $O(1/\rho^2)$

**Step 3**: Establish k-uniform Hessian bounds for localized moments
- **Key Lemma**: Telescoping identity $\sum_{j \in A_k} \nabla^2 w_{ij} = 0$ (follows from differentiating $\sum_j w_{ij} = 1$ twice)
- **Application to mean**:
  $$\nabla^2 \mu_\rho = \sum_j \nabla^2 w_{ij} \cdot d(x_j) = \sum_j \nabla^2 w_{ij} \cdot (d(x_j) - \mu_\rho)$$
  Taking norms with $|d(x_j) - \mu_\rho| \le \text{diam}(d)$ and only $k_{\text{eff}} = O(1)$ walkers in ρ-neighborhood:
  $$\|\nabla^2 \mu_\rho\| \le C_w(\rho) \cdot \text{diam}(d) \cdot k_{\text{eff}} = O(1/\rho^2)$$
  **k-independent** due to telescoping!

- **Application to variance**: Similar telescoping for $\nabla^2 \sigma^2_\rho$, yields $\|\nabla^2 \sigma^2_\rho\| = O(1/\rho^2)$, k-uniform

**Step 4**: Bound derivatives of $\sigma'_\rho = \sigma'_{\text{reg}}(\sigma^2_\rho)$ via chain rule
- **First derivative**: $\nabla \sigma'_\rho = (\sigma'_{\text{reg}})'(\sigma^2_\rho) \nabla \sigma^2_\rho$ with $\|\nabla \sigma'_\rho\| = O(1/\rho)$
- **Second derivative**: $\nabla^2 \sigma'_\rho = (\sigma'_{\text{reg}})''(\sigma^2_\rho) (\nabla \sigma^2_\rho)^{\otimes 2} + (\sigma'_{\text{reg}})'(\sigma^2_\rho) \nabla^2 \sigma^2_\rho$
  with $\|\nabla^2 \sigma'_\rho\| = O(1/\rho^2)$
- **Denominator stability**: $\sigma'_\rho \ge \epsilon_\sigma > 0$ prevents division by zero in all quotient terms

**Step 5**: Assemble all bounds
- Bound each of the four terms in $\nabla^2 Z_\rho$ using Steps 3-4:
  - Term 1: $(1/\sigma'_\rho)(\nabla^2 d - \nabla^2 \mu_\rho) = O(1) \cdot [O(1) + O(1/\rho^2)] = O(1/\rho^2)$
  - Term 2: $(1/(\sigma'_\rho)^2)[O(1/\rho) \otimes O(1/\rho)] = O(1) \cdot O(1/\rho^2) = O(1/\rho^2)$
  - Term 3: $(O(1)/(\sigma'_\rho)^2) O(1/\rho^2) = O(1/\rho^2)$
  - Term 4: $(O(1)/(\sigma'_\rho)^3)[O(1/\rho) \otimes O(1/\rho)] = O(1) \cdot O(1/\rho^2) = O(1/\rho^2)$
- Therefore $\|\nabla^2 Z_\rho\| = O(1/\rho^2)$, k-uniform
- From Step 1 with $\|\nabla Z_\rho\| = O(1/\rho)$ (C¹ theorem):
  $$\|\nabla^2 V_{\text{fit}}\| \le O(1) \cdot O(1/\rho^2) + O(1) \cdot O(1/\rho^2) = O(1/\rho^2)$$

**Conclusion**: $H_{\max}(\rho) = O(1/\rho^2)$, k-uniform and N-uniform. □

---

## Technical Lemmas Required

The proof requires several supporting lemmas:

**Lemma A** (Weight derivative bounds):
- Statement: For Gaussian kernel $K_\rho(r) = \exp(-r^2/(2\rho^2))$, the normalized weights satisfy:
  $$\|\nabla w_{ij}\| \le \frac{2C_{\nabla K}(\rho)}{\rho}, \quad \|\nabla^2 w_{ij}\| \le C_w(\rho) = O(1/\rho^2)$$
- Difficulty: **EASY** - Standard Gaussian derivative bounds using Hermite polynomials
- Source: Gaussian kernel axiom, Hermite bounds in framework

**Lemma B** (Telescoping identities):
- Statement: $\sum_{j \in A_k} \nabla^m w_{ij}(\rho) = 0$ for all $m \ge 1$
- Proof: Differentiate $\sum_{j \in A_k} w_{ij}(\rho) = 1$ m times, noting the right side is constant
- Difficulty: **EASY** - Trivial from normalization property
- Critical for: k-uniformity throughout

**Lemma C** (k-uniform Hessian of localized mean):
- Statement: $\|\nabla^2 \mu_\rho\| \le C_{\mu^2}(\rho) = O(1/\rho^2)$, independent of k
- Proof: Use Lemma B to convert $\sum_j \nabla^2 w_{ij} \cdot d(x_j)$ to centered sum, bound by $O(1/\rho^2)$
- Difficulty: **MEDIUM** - Requires careful telescoping application
- Key insight: Centered terms $(d(x_j) - \mu_\rho)$ uniformly bounded, only $k_{\text{eff}}$ walkers contribute

**Lemma D** (k-uniform Hessian of localized variance):
- Statement: $\|\nabla^2 \sigma^2_\rho\| \le C_{V^2}(\rho) = O(1/\rho^2)$, independent of k
- Proof: Apply Leibniz rule to $\sigma^2_\rho = \sum_j w_{ij}(d(x_j) - \mu_\rho)^2$, use telescoping on highest-order term
- Difficulty: **MEDIUM** - Product rule generates multiple terms, need to track all
- Key insight: Similar to Lemma C, telescoping prevents k-accumulation

**Lemma E** (Chain rule bounds for regularized std dev):
- Statement: $\|\nabla \sigma'_\rho\| = O(1/\rho)$, $\|\nabla^2 \sigma'_\rho\| = O(1/\rho^2)$
- Proof: Chain rule on $\sigma'_\rho = \sigma'_{\text{reg}}(\sigma^2_\rho)$ with $\sigma'_{\text{reg}} \in C^\infty$
- Difficulty: **EASY** - Standard composition calculus
- Critical property: $\sigma'_{\text{reg}} \ge \epsilon_\sigma > 0$ ensures bounded denominators

---

## Difficulty Assessment

**Overall Difficulty**: **MEDIUM**

**Breakdown**:
- **Calculus machinery**: Standard (chain, product, quotient rules) - **LOW**
- **Telescoping mechanism**: Requires insight but straightforward once identified - **MEDIUM**
- **k-uniformity tracking**: Central challenge, needs careful accounting - **MEDIUM**
- **ρ-scaling aggregation**: Moderate bookkeeping across four term types - **MEDIUM**
- **Denominator stability**: Handled by regularization axiom - **LOW**

**Why not HIGH**:
- All required calculus is multivariate but standard (no exotic differential geometry)
- Telescoping pattern is established in C¹ theorem (can import technique)
- Source document provides explicit formulas and bounds to verify against
- Supporting lemmas (A-E) are all straightforward to prove in detail

**Why not LOW**:
- Four-term quotient rule expansion requires careful tracking
- k-uniformity is subtle (naive approach gives k-dependent bound)
- Must verify telescoping works at second-order (not just first-order)
- Aggregating different ρ-scalings requires systematic analysis

---

## Expansion Time Estimate

**Supporting Lemmas** (A-E): 3-4 hours
- Lemma A: 30 min (Gaussian calculus)
- Lemma B: 15 min (trivial from normalization)
- Lemma C: 1-1.5 hours (telescoping application)
- Lemma D: 1-1.5 hours (variance Hessian with Leibniz)
- Lemma E: 30 min (chain rule)

**Main Proof Steps** (1-5): 4-5 hours
- Step 1 (chain rule): 30 min (standard)
- Step 2 (quotient rule expansion): 1.5 hours (four terms, tensor algebra)
- Step 3 (moment Hessians): 1 hour (apply Lemmas C-D)
- Step 4 (σ' derivatives): 30 min (apply Lemma E)
- Step 5 (assembly): 1-1.5 hours (bound all four quotient terms, verify k-uniformity)

**Rigor and Cross-Validation**: 3-4 hours
- Interchange of differentiation and summation: 1 hour
- Compactness arguments: 30 min
- Edge cases (k=1, ρ→0, ρ→∞): 1 hour
- Framework cross-validation: 1 hour
- Constant tracking audit: 30-60 min

**Total**: **10-13 hours** for complete detailed proof with full rigor

**Priority**: **HIGH** - This theorem is:
1. Base case for C³, C⁴, and C∞ regularity induction
2. Critical for BAOAB integrator stability analysis
3. Prerequisite for uniform ellipticity theorem (Chapter 4)
4. Enables adaptive diffusion tensor construction

---

## Framework Dependencies

**Required Theorems**:
- {prf:ref}`thm-c1-regularity` (11_geometric_gas.md § A.3): C¹ bounds on $V_{\text{fit}}$, provides $\|\nabla Z_\rho\| = O(1/\rho)$

**Required Definitions**:
- Gaussian localization kernel $K_\rho(r)$ with Hermite derivative bounds
- Normalized localization weights $w_{ij}(\rho)$ with $\sum_j w_{ij} = 1$
- Localized moments $\mu_\rho$, $\sigma^2_\rho$
- Regularized standard deviation $\sigma'_{\text{reg}}$ with $\sigma'_{\text{reg}} \ge \epsilon_\sigma > 0$
- Z-score $Z_\rho = (d(x_i) - \mu_\rho)/\sigma'_\rho$
- Fitness potential $V_{\text{fit}} = g_A(Z_\rho)$

**Required Axioms**:
- Compact state space $\mathcal{X}$ with $C^\infty$ boundary
- Measurement $d \in C^\infty(\mathcal{X})$ with bounded derivatives
- Rescale function $g_A \in C^\infty$ with bounded derivatives $L_{g_A}$, $L_{g''_A}$
- Regularization $\sigma'_{\text{reg}} \in C^\infty$ with $\sigma'_{\text{reg}} \ge \epsilon_\sigma > 0$

**No Circular Dependencies**: C² proof uses C¹ as prerequisite (not conclusion). C¹ proven independently in § A.3.

---

## Review Protocol

**Review Strategy**: Single-strategist (GPT-5/Codex only)

**Rationale**: Gemini 2.5 Pro has MCP issues preventing its use. While this reduces cross-validation confidence, the proof strategy is:
1. Well-established (matches existing proof in 11_geometric_gas.md § A.4)
2. Verified against source document line-by-line (lines 2930-3029)
3. Consistent with C¹ theorem technique (telescoping mechanism)

**Mitigation**:
- All steps cross-checked against source proof
- Framework dependencies verified in glossary.md
- Logical soundness audit performed
- Will flag as lower confidence in final output

**Recommendation**: Re-run with dual verification (Gemini + Codex) when Gemini MCP is restored

---

## Return Path

**Sketch Output**: /home/guillem/fragile/docs/source/2_geometric_gas/sketcher/sketch_thm_c2_established_cinf.md

**Next Steps**:
1. Submit sketch to GPT-5 (Codex) for review
2. Incorporate Codex feedback
3. Mark as ready for full expansion (Phase 2: Theorem Prover)
4. Flag for dual re-review when Gemini available

**Proof Sketch Status**: ✅ COMPLETE (single-strategist, medium confidence)
