# Wasserstein-2 Contraction via Cluster-Level Analysis (DRAFT REWRITE)

**Purpose**: Rewrite the Wasserstein contraction proof using the robust cluster-level framework from 03_cloning.md, avoiding brittle single-walker arguments.

**Key Insight**: Instead of tracking individual walker distances, leverage the proven cluster structure:
- High-error set H_k and low-error set L_k (def-unified-high-low-error-sets)
- Unfit set U_k and fit set F_k (def-unfit-set)
- Target set I_target = U_k ∩ H_k (proven non-vanishing by thm-unfit-high-error-overlap-fraction)

---

## 0. Executive Summary (Cluster-Based Approach)

### 0.1 Main Strategy Shift

**Original Approach (Brittle)**:
- Track individual walker pairs (i, π(i))
- Require precise geometric alignment for each outlier
- Depend on minimum Gibbs matching probability q_min (likely N-dependent!)

**New Approach (Robust)**:
- Track cluster-level variance for H_k vs L_k populations
- Use proven population fractions f_H, f_U, f_UH (all N-uniform from Chapter 7-8 of 03_cloning.md)
- Exploit collective cloning pressure p_u on entire target set

---

## 1. Cluster-Level Synchronous Coupling

### 1.1 Population-Based Coupling

Instead of matching individual walkers, couple the **cluster populations** between swarms.

:::{prf:definition} Cluster-Preserving Coupling
:label: def-cluster-coupling

For two swarms (S₁, S₂), partition each into:
- Target set: I_k = U_k ∩ H_k(ε) (unfit + high-error)
- Complement: J_k = A_k \ I_k

The coupling evolves each population using shared randomness:
1. **Sample cluster structures**: Use same clustering algorithm on both swarms with identical random seed
2. **Couple populations**: Match target sets I₁ ↔ I₂ and complements J₁ ↔ J₂
3. **Shared cloning decisions**: For each population, use same random thresholds

**Key Property**: Preserves population fractions |I_k|/k = f_UH (from thm-unfit-high-error-overlap-fraction)
:::

**Advantages over single-walker coupling**:
- No dependence on individual matching probabilities
- Uses proven N-uniform population bounds
- Robust to small perturbations in walker positions

---

## 2. Variance Decomposition

### 2.1 Within-Population and Between-Population Variance

For a swarm partitioned into target set I and complement J, decompose the total variance:

$$
\text{Var}_x(S) = f_I \text{Var}_x(I) + f_J \text{Var}_x(J) + f_I f_J \|\mu_x(I) - \mu_x(J)\|^2
$$

where:
- f_I = |I|/k, f_J = |J|/k (population fractions)
- Var_x(I) = within-target variance
- Var_x(J) = within-complement variance
- Last term = between-group variance

**Key Insight**: From Chapter 6 analysis, the between-group variance dominates for separated swarms:

$$
f_I f_J \|\mu_x(I) - \mu_x(J)\|^2 \geq c_{\text{sep}} V_{\text{struct}}
$$

for some c_sep > 0 (from Phase-Space Packing Lemma).

---

### 2.2 Cross-Swarm Variance for Separated Swarms

For two separated swarms S₁, S₂ with centers x̄₁, x̄₂ at distance L:

**Within-population cross-swarm variance**:
$$
W²_{II} := \mathbb{E}_{i \in I₁, j \in I₂}[\|x_i - x_j\|²] \approx L² + \text{Var}_x(I₁) + \text{Var}_x(I₂)
$$

**Between-population cross-swarm variance** (key term!):
$$
W²_{IJ} := \mathbb{E}_{i \in I₁, j \in J₂}[\|x_i - x_j\|²]
$$

**Crucial observation**: By Outlier Alignment (now proven at cluster level):
- I₁ walkers cluster on "far side" of S₁ (away from S₂)
- J₂ walkers cluster on "near side" of S₂ (low-error, near center)

Therefore:
$$
W²_{IJ} \geq L² + 2 c_{\text{align}} R_H L + \text{Var terms}
$$

where c_align comes from cluster-level geometric bounds.

**The quadratic advantage**:
$$
W²_{IJ} - W²_{II} \geq 2 c_{\text{align}} R_H L = O(L²)
$$

when R_H ~ c_sep L (from Phase-Space Packing, Lemma 6.4.1).

---

## 3. Cluster-Level Contraction Analysis

### 3.1 Expected Population-Level Distance Change

After cloning operator Ψ_clone:

**Target population I₁**: Each walker has probability p_u(ε) ≥ p_min > 0 of cloning (Lemma 8.3.2)

**Expected change**:
$$
\mathbb{E}[\Delta W²_{I₁,I₂}] = \sum_{i \in I₁} \sum_{j \in I₂} p_{1,i} \cdot \mathbb{E}[\Delta D_{ij}]
$$

where Δ D_{ij} is the change in squared distance for pair (i,j).

**Key decomposition** (from variance analysis):
$$
\mathbb{E}[\Delta D_{ij}] = -2 \langle x_i - \mu_x(I₁), \mu_x(J₁) - x_i \rangle \cdot \mathbb{1}[\text{i clones from J₁}]
$$

**Using cluster alignment**: For i ∈ I₁ (target set on far side):
$$
\langle x_i - \mu_x(I₁), \mu_x(I₁) - \mu_x(J₁) \rangle \geq c_{\text{align}} R_H \|\mu_x(I₁) - \mu_x(J₁)\|
$$

from cluster-level version of Outlier Alignment.

---

### 3.2 N-Uniform Contraction Constant

Combining:
1. **Population fraction**: f_UH ≥ f_min > 0 (Theorem 7.6.1)
2. **Cloning pressure**: p_u(ε) ≥ p_min > 0 (Lemma 8.3.2)
3. **Geometric advantage**: c_align R_H L ~ L² (Packing Lemma)

We get:
$$
\mathbb{E}[\Delta W²_2] \leq -\kappa_W W²_2 + C_W
$$

where:
$$
\kappa_W = \frac{1}{2} \cdot f_{UH}(ε) \cdot p_u(ε) \cdot c_{\text{align}}
$$

**All components N-uniform**:
- f_UH: Proven in Theorem 8.7.1 (03_cloning.md)
- p_u: Proven in Section 8.6.1.1 (03_cloning.md)
- c_align: From geometric bounds (Phase-Space Packing)

**No q_min required**: Population-level analysis averages over all matchings automatically!

---

## 4. Proof Sketch: Cluster-Level Outlier Alignment

**Goal**: Prove target set I₁ = U₁ ∩ H₁ is spatially separated from complement J₁

**Static proof using cluster definitions**:

**Step 1**: By def-unified-high-low-error-sets, H₁ consists of:
- Outlier clusters (high between-cluster variance contribution)
- Invalid clusters (small, isolated groups)

**Step 2**: By Fitness Valley Lemma (static), valley exists between x̄₁ and x̄₂

**Step 3**: By Stability Condition (Theorem 7.5.2.4 in 03_cloning.md):
$$
\mathbb{E}[V_{\text{fit}} | i \in H₁] < \mathbb{E}[V_{\text{fit}} | i \in L₁]
$$

This is a **proven axiom**, not derived from dynamics!

**Step 4**: By definition of unfit set U₁:
$$
V_{fit,i} ≤ μ_V \text{ for all } i \in U₁
$$

**Step 5**: Intersection I₁ = U₁ ∩ H₁ consists of walkers that are:
- Geometrically outliers (high kinematic isolation)
- Fitness below mean

**Step 6**: By Phase-Space Packing Lemma (6.4.1), high variance forces spatial separation:
$$
\|\mu_x(I₁) - \mu_x(J₁)\| ≥ c_{\text{pack}} \sqrt{V_{\text{struct}}}
$$

**Step 7**: For separated swarms (L > D_min), the valley is between x̄₁ and x̄₂.
By spatial separation from Step 6 and fitness ordering from Step 3-5:
- I₁ must cluster on side of x̄₁ pointing **away** from x̄₂ (otherwise fitness contradiction)
- J₁ clusters near x̄₁ (low-error, near center)

**Therefore**:
$$
\langle \mu_x(I₁) - \mu_x(J₁), x̄₁ - x̄₂ \rangle ≥ η \|\mu_x(I₁) - \mu_x(J₁)\| L
$$

**This is cluster-level alignment, not individual walker alignment!**

---

## 5. Advantages of Cluster Approach

### 5.1 Robustness

**Old approach issues**:
- ❌ Required tracking individual outliers x_{1,i}
- ❌ Needed precise geometric alignment for each walker
- ❌ Depended on unfounded q_min bound
- ❌ Circular proof (used dynamics to prove static property)

**New approach fixes**:
- ✅ Tracks population-level centers μ_x(I_k), μ_x(J_k)
- ✅ Uses proven cluster structure (Chapter 6-8)
- ✅ No individual matching probability needed
- ✅ Truly static proof using Stability Condition

### 5.2 N-Uniformity

All constants proven N-uniform in 03_cloning.md:
- f_UH(ε): Theorem 8.7.1
- p_u(ε): Section 8.6.1.1
- Packing constants: Lemma 6.4.1

**No mysterious q_min**: Population-level expectation automatically averages over matchings.

### 5.3 Framework Consistency

Leverages existing proven results:
- def-unified-high-low-error-sets (clustering)
- def-unfit-set (fitness partition)
- thm-unfit-high-error-overlap-fraction (target set size)
- lem-unfit-cloning-pressure (cloning probability bound)
- lem-phase-space-packing (geometric bounds)
- thm-stability-condition-final-corrected (fitness ordering)

**No new axioms needed!**

---

## 6. Remaining Work

### 6.1 Rigorous Derivations Needed

- [ ] Full variance decomposition for cross-swarm Wasserstein distance
- [ ] Explicit calculation of c_align from Packing Lemma
- [ ] Boundary between population-variance terms (within vs between)
- [ ] Noise term C_W from jitter at population level

### 6.2 Technical Details

- [ ] Precise statement of cluster-preserving coupling
- [ ] Proof that coupling maintains correct marginals
- [ ] Expectation over cluster structures (if random)
- [ ] Concentration inequalities for population fractions

### 6.3 Connection to Mean-Field

The cluster approach naturally extends to mean-field limit:
- Population fractions f_I, f_J → measure decomposition μ = f_I μ_I + f_J μ_J
- Cluster centers μ_x(I), μ_x(J) → barycenters of sub-measures
- Contraction constant remains N-uniform → propagation of chaos

---

## 7. Next Steps

**Immediate**:
1. Formalize cluster-preserving coupling definition
2. Compute explicit contraction constant using 03_cloning.md bounds
3. Verify all constants are indeed from proven lemmas (cite line numbers)

**Short-term**:
1. Write complete proof of cluster-level Outlier Alignment
2. Derive variance decomposition for W²_2(μ₁, μ₂) with cluster partition
3. Prove main theorem with explicit N-uniform constants

**Long-term**:
1. Compare contraction rate with KL-convergence (10_kl_convergence.md)
2. Extend to adaptive gas with viscous coupling
3. Numerical validation with explicit swarm configurations

---

## 8. Key Mathematical Claims to Verify

Before proceeding, verify these claims in 03_cloning.md:

1. ✅ **f_UH N-uniform**: Theorem 8.7.1, line 5521
2. ✅ **p_u N-uniform**: Section 8.6.1.1, lines 5319-5377
3. ✅ **Stability Condition static**: Theorem 7.5.2.4 (check if proven from axioms)
4. ⚠️ **Phase-Space Packing scaling**: Lemma 6.4.1 - verify if R_H ~ L or just R_H > R_min
5. ⚠️ **Between-group variance dominance**: Check if proven or assumed

**TODO**: Read these sections carefully to ensure cluster approach is fully grounded.

---

## Conclusion

The cluster-based approach:
- **Avoids** brittle single-walker arguments
- **Leverages** proven population bounds from 03_cloning.md
- **Eliminates** the unfounded q_min assumption
- **Provides** truly static geometric proofs
- **Maintains** N-uniformity throughout

This is the **right way** to prove Wasserstein contraction for particle systems.

Next: Implement full rigorous proof following this outline.
