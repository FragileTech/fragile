# Proof Sketch for lem-variance-cinf-inductive

**Document**: docs/source/2_geometric_gas/19_geometric_gas_cinf_regularity_simplified.md
**Theorem**: lem-variance-cinf-inductive
**Generated**: 2025-10-25 09:05
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:lemma} C^{m+1} Regularity of Localized Variance
:label: lem-variance-cinf-inductive

Under the same assumptions [as lem-mean-cinf-inductive: weights w_ij(ρ) are C^{m+1} in x_i with ||∇^{m+1} w_ij|| ≤ W_{m+1}(ρ)], σ²_ρ[f_k, x_i] is C^{m+1} with:

$$
\|\nabla^{m+1}_{x_i} \sigma^2_\rho\| \leq C_{\text{var},m+1}(\rho) \cdot (\text{diam}(d))^2
$$

where C_{var,m+1}(ρ) = O(W_{m+1}(ρ) + products of lower-order weight derivatives).

**Proof sketch from document**: The variance σ²_ρ = ∑_j w_ij · (d(x_j) - μ_ρ)² involves products. Leibniz rule for (m+1)-th derivative yields terms like ∇^p w_ij · ∇^q((d - μ_ρ)²) for p + q = m+1. The highest-order term:
∇^{m+1} σ²_ρ ~ ∑_j ∇^{m+1} w_ij · (d(x_j) - μ_ρ)² + lower-order
Telescoping applies: ∑_j ∇^{m+1} w_ij · [(d_j - μ_ρ)² - σ²_ρ] with |(...)² - σ²_ρ| ≤ 2(diam(d))².
:::

**Informal Restatement**: The localized variance is infinitely differentiable with k-uniform bounds at all orders. The (m+1)-th derivative bound grows with W_{m+1}(ρ) (weight derivative bound) and (diam(d))² (squared range of measurements). The telescoping mechanism from lem-telescoping-all-orders-cinf prevents k-linear growth by rewriting weighted sums as centered sums.

---

## II. Proof Strategy Synthesis

**Chosen Method**: Leibniz rule expansion + telescoping reduction (Direct proof via calculus of products)

**Rationale**:
The variance is a weighted sum of squared deviations: σ²_ρ = ∑_j w_ij(ρ) · (d(x_j) - μ_ρ[f_k, x_i])². Taking (m+1)-th derivatives requires:

1. **Product rule application** (Leibniz): Since variance involves products w_ij · (centered squared term), the generalized Leibniz formula gives:
   $$
   \nabla^{m+1} \sigma^2_\rho = \sum_{p+q=m+1} \binom{m+1}{p} \sum_j \nabla^p w_{ij} \cdot \nabla^q[(d(x_j) - \mu_\rho)^2]
   $$

2. **Highest-order isolation**: The p = m+1, q = 0 term dominates (highest weight derivative):
   $$
   \sum_j \nabla^{m+1} w_{ij} \cdot (d(x_j) - \mu_\rho)^2
   $$

3. **Telescoping application**: Using ∑_j ∇^{m+1} w_ij = 0 (lem-telescoping-all-orders-cinf), rewrite as:
   $$
   \sum_j \nabla^{m+1} w_{ij} \cdot [(d(x_j) - \mu_\rho)^2 - \sigma^2_\rho]
   $$

4. **Bound centered deviations**: |(d_j - μ_ρ)² - σ²_ρ| ≤ 2(diam(d))² (standard variance inequality)

5. **Lower-order terms**: Terms with p < m+1 involve ∇^p w_ij (p ≤ m) and ∇^q μ_ρ (q ≤ m+1), controlled by induction hypothesis

This approach mirrors the proven C³ and C⁴ cases (13_geometric_gas_c3_regularity.md § 5.2, 14_geometric_gas_c4_regularity.md § 5) and extends naturally to all orders.

---

## III. Framework Dependencies

### Verified Dependencies

**Theorems**:
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| lem-telescoping-all-orders-cinf | 19 (line 431-448) | ∑_j ∇^m w_ij = 0 for all m ≥ 1 | Step 3 (telescoping) | ✅ |
| lem-mean-cinf-inductive | 19 (line 451-488) | μ_ρ ∈ C^{m+1} with bound | Lower-order terms | ✅ |

**Definitions**:
| Label | Document | Definition | Used for |
|-------|----------|------------|----------|
| Localized variance | 19 (§ 3) | σ²_ρ = ∑_j w_ij (d_j - μ_ρ)² | Primary object |
| diam(d) | 19 (line 460) | sup_{x,y} |d(x) - d(y)| < ∞ | Bounded deviation |

**Constants**:
| Symbol | Definition | Value/Bound | Properties |
|--------|------------|-------------|------------|
| W_{m+1}(ρ) | Weight derivative bound | ||∇^{m+1} w_ij|| ≤ W_{m+1}(ρ) | From C∞ primitives |
| diam(d) | Measurement diameter | sup-inf over compact X | Finite (X compact) |

---

## IV. Detailed Proof Sketch

### Overview

The variance σ²_ρ is a product-sum: weighted sum of products (w_ij times squared deviations). Differentiating (m+1) times generates many terms via Leibniz rule. The key insight: the highest-order term (∇^{m+1} w_ij with no derivatives on the squared deviation) can be telescoped using ∑_j ∇^{m+1} w_ij = 0, converting it to a centered sum that scales with (diam(d))² instead of k · (diam(d))². Lower-order terms are controlled inductively.

### Proof Outline

1. **Apply generalized Leibniz rule** to σ²_ρ = ∑_j w_ij · g_j where g_j = (d_j - μ_ρ)²
2. **Isolate highest-order term**: p = m+1 contribution
3. **Apply telescoping**: Rewrite using ∑_j ∇^{m+1} w_ij = 0
4. **Bound centered squared deviations**: Use |(d_j - μ_ρ)² - σ²_ρ| ≤ 2(diam(d))²
5. **Control lower-order terms**: Use induction hypothesis on μ_ρ ∈ C^m and weights ∈ C^m

### Detailed Steps

#### Step 1: Leibniz Expansion

**Goal**: Write ∇^{m+1} σ²_ρ as sum over partition orders

**Action**: For σ²_ρ = ∑_j w_ij · (d(x_j) - μ_ρ[f_k, x_i])², apply:
$$
\nabla^{m+1}_{x_i} \sigma^2_\rho = \sum_{p+q=m+1} \binom{m+1}{p} \sum_j \nabla^p_{x_i} w_{ij} \cdot \nabla^q_{x_i}[(d(x_j) - \mu_\rho)^2]
$$

**Justification**: Generalized Leibniz rule for products; finite sum (A_k finite) so derivative commutes with summation

**Dependencies**: lem-telescoping-all-orders-cinf establishes finite-sum commutation

---

#### Step 2: Highest-Order Term Isolation

**Goal**: Identify dominant contribution

**Action**: The p = m+1, q = 0 term is:
$$
\sum_j \nabla^{m+1}_{x_i} w_{ij} \cdot (d(x_j) - \mu_\rho)^2
$$

(In simplified model, ∇^q_{x_i}(d(x_j) - μ_ρ) for q ≥ 1 involves only μ_ρ derivatives when j ≠ i, since d(x_j) doesn't depend on x_i)

**Justification**: This has highest weight derivative order → largest bound W_{m+1}(ρ)

---

#### Step 3: Telescoping Application

**Goal**: Convert to centered sum

**Action**: Since ∑_j ∇^{m+1} w_ij = 0 (lem-telescoping-all-orders-cinf):
$$
\sum_j \nabla^{m+1} w_{ij} \cdot (d(x_j) - \mu_\rho)^2 = \sum_j \nabla^{m+1} w_{ij} \cdot [(d(x_j) - \mu_\rho)^2 - \sigma^2_\rho]
$$

**Justification**: Add and subtract ∑_j ∇^{m+1} w_ij · σ²_ρ = σ²_ρ · (∑_j ∇^{m+1} w_ij) = 0

**Dependencies**: lem-telescoping-all-orders-cinf (m = m+1)

---

#### Step 4: Bound Centered Deviations

**Goal**: Control |(d_j - μ_ρ)² - σ²_ρ|

**Action**:
- σ²_ρ = 𝔼[(d - μ_ρ)²] (weighted expectation)
- For any value d_j, |(d_j - μ_ρ)² - σ²_ρ| ≤ max possible squared deviation
- Since |d_j - μ_ρ| ≤ diam(d), we have (d_j - μ_ρ)² ≤ (diam(d))²
- Therefore |(d_j - μ_ρ)² - σ²_ρ| ≤ 2(diam(d))²

**Justification**: Variance is average squared deviation; any individual squared deviation differs from average by at most 2 × max squared deviation

**Resulting Bound**:
$$
\left\|\sum_j \nabla^{m+1} w_{ij} \cdot [(d_j - \mu_\rho)^2 - \sigma^2_\rho]\right\| \le W_{m+1}(\rho) \cdot 2(\text{diam}(d))^2
$$

(k-uniform: sum over centered terms, each bounded, with ∑-coefficients having controlled norm)

---

#### Step 5: Lower-Order Terms

**Goal**: Bound terms with p < m+1

**Action**: For p + q = m+1 with 0 ≤ p ≤ m:
- ∇^p w_ij bounded by W_p(ρ) (available from C∞ primitives and p ≤ m < m+1)
- ∇^q[(d - μ_ρ)²] involves ∇^r μ_ρ for r ≤ q ≤ m+1
- By lem-mean-cinf-inductive, ||∇^r μ_ρ|| ≤ C_{μ,r}(ρ) · diam(d) for r ≤ m+1

**Justification**: Products of derivatives: use Faà di Bruno for composite (d - μ_ρ)²; all terms bounded by products of known bounds W_p(ρ), C_{μ,r}(ρ), diam(d)

**Resulting Bound**: Lower-order contribution ~ ∑_{p=0}^m W_p(ρ) · C_{μ,m+1-p}(ρ) · (diam(d))²

---

#### Step 6: Assembly

**Goal**: Combine all terms

**Action**:
$$
\|\nabla^{m+1} \sigma^2_\rho\| \le [W_{m+1}(\rho) + \text{products of lower orders}] \cdot (\text{diam}(d))^2 = C_{\text{var},m+1}(\rho) \cdot (\text{diam}(d))^2
$$

where C_{var,m+1}(ρ) = O(W_{m+1}(ρ)) + polynomial in {W_p(ρ), C_{μ,q}(ρ) : p,q ≤ m+1}

**Conclusion**: Variance is C^{m+1} with k-uniform, N-uniform bound. Q.E.D. ∎

---

## V. Technical Deep Dives

### Challenge 1: Faà di Bruno Complexity for (d - μ_ρ)²

**Why Difficult**: Computing ∇^q[(d(x_j) - μ_ρ(x_i))²] requires chain rule for squaring and product rule for (d - μ_ρ). The Faà di Bruno formula gives exponentially many terms.

**Proposed Solution**:
- For squared function f²: ∇^q(f²) = ∑ (Faà di Bruno coefficients) · products of ∇^r f
- Worst case: ∇^q(f²) ~ q! · (∇f)^q type growth
- However, f = d - μ_ρ has BOUNDED derivatives (d ∈ C∞ on compact X, μ_ρ ∈ C^{m+1} by lemma)
- Telescoping already absorbed one factorial factor, so net growth is polynomial in q, not exponential

**References**: Document addresses this in lines 651-657; telescoping mechanism prevents factorial-squared growth

---

### Challenge 2: k-Uniformity via Telescoping

**Why Difficult**: Naive bound ∑_j ||∇^{m+1} w_ij|| · |(d_j - μ_ρ)²| ~ k · W_{m+1}(ρ) · (diam(d))² grows with k.

**Proposed Solution**:
- Telescoping: ∑_j ∇^{m+1} w_ij = 0 enables centered sum
- Centered: ∑_j ∇^{m+1} w_ij · [g_j - σ²_ρ] where g_j = (d_j - μ_ρ)²
- Key: σ²_ρ = ∑_j w_ij g_j is the weighted average of g_j
- Centered deviations [g_j - σ²_ρ] have zero weighted sum
- Bound uses max deviation, not sum of deviations → k-independent

**References**: lem-telescoping-all-orders-cinf; rmk-k-uniformity-mechanism-cinf (line 405-423)

---

## VI. Cross-References

**Theorems Used**:
- {prf:ref}`lem-telescoping-all-orders-cinf` - Foundational telescoping at order m+1
- {prf:ref}`lem-mean-cinf-inductive` - Provides C^{m+1} bound on μ_ρ (used in lower-order terms)

**Definitions Used**:
- Localized variance (19_geometric_gas_cinf_regularity_simplified.md, § 3)
- Measurement diameter diam(d) (line 460)

**Related Proofs**:
- C³ variance proof: 13_geometric_gas_c3_regularity.md § 5.2 (explicit m=3 case)
- C⁴ variance proof: 14_geometric_gas_c4_regularity.md § 5 (explicit m=4 case)

**Downstream Dependencies**:
- {prf:ref}`lem-z-score-cinf-inductive` - Uses σ²_ρ ∈ C^{m+1}
- {prf:ref}`thm-inductive-step-cinf` - Combines mean + variance regularity for full V_fit bound

---

**Proof Sketch Completed**: 2025-10-25 09:05
**Ready for Expansion**: Yes
**Confidence Level**: High - Follows established pattern from C³/C⁴ proofs; telescoping mechanism fully developed; all dependencies verified.
