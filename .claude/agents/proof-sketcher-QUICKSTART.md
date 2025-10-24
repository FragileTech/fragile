# Proof Sketcher Agent - Quick Start Guide

## Simplest Usage (Copy-Paste Ready)

### Single Theorem Sketch

Just paste this into Claude:

```
Load the proof-sketcher agent from .claude/agents/proof-sketcher.md
and sketch proof for:

Document: docs/source/1_euclidean_gas/09_kl_convergence.md
Theorem: thm-kl-convergence-euclidean

Use thorough depth.
```

### Multiple Theorems in Same Document

```
Load proof-sketcher agent.

Sketch proofs for: docs/source/1_euclidean_gas/06_convergence.md
Focus on: Foster-Lyapunov main theorem and drift condition lemmas
Depth: thorough
```

### Complete Document (All Theorems)

```
Load proof-sketcher agent.

Sketch all proofs for: docs/source/1_euclidean_gas/08_propagation_chaos.md
Depth: exhaustive
```

---

## What Happens

1. **Agent reads the document** strategically (extracts theorem statements)
2. **Identifies dependencies** by searching for `{prf:ref}` and checking glossary
3. **Submits identical prompts** to Gemini 2.5 Pro + GPT-5 Pro in parallel
4. **Waits for both proof strategies** to be returned
5. **Critically compares** approaches:
   - Identifies consensus (both agree on approach)
   - Identifies contradictions (different methods)
   - Cross-validates against framework docs
6. **Synthesizes optimal strategy** with evidence-based justification
7. **Generates detailed proof sketch** with 10 sections:
   - Theorem statement
   - Strategy comparison (Gemini vs GPT-5 vs Claude synthesis)
   - Framework dependencies (verified against glossary)
   - Detailed step-by-step sketch
   - Technical deep dives (1-3 hardest parts)
   - Validation checklist
   - Alternative approaches
   - Open questions
   - Expansion roadmap
   - Cross-references
8. **Writes sketch to file**: `sketcher/sketch_{timestamp}_proof_{filename}.md`

---

## Expected Output Format

You'll receive a proof sketch like this:

```markdown
# Proof Sketch for thm-kl-convergence-euclidean

**Document**: docs/source/1_euclidean_gas/09_kl_convergence.md
**Generated**: 2025-10-24 15:30
**Agent**: Proof Sketcher v1.0

---

## I. Theorem Statement

:::{prf:theorem} N-Particle Exponential Convergence via N-Uniform LSI
:label: thm-kl-convergence-euclidean

For Euclidean Gas with parameters satisfying the Kinetic Dominance Condition,
there exists an N-uniform LSI constant λ_LSI > 0 such that:

$$
D_{KL}(\rho_t^N || \rho_\infty^N) \leq e^{-\lambda_{LSI} t} D_{KL}(\rho_0^N || \rho_\infty^N)
$$

for all N ≥ 1 and all initial distributions ρ_0^N.
:::

---

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach
**Method**: Lyapunov method via entropy production
**Key Steps**:
1. Define H(ρ) = D_KL(ρ || ρ_∞) as Lyapunov function
2. Compute dH/dt via generator action
3. Decompose into velocity diffusion (dissipation) + cloning (expansion/contraction)
4. Apply synergistic dissipation lemma (doc-03, Lemma 8.2)
5. Show dissipation dominates expansion under Kinetic Dominance
6. Conclude exponential decay with rate λ_LSI

**Strengths**:
- Direct proof of LSI (no indirect inequalities)
- Explicit constant tracking ensures N-uniformity
- Framework already provides all needed machinery

**Weaknesses**:
- Fisher information calculation is technically involved
- Requires careful tracking of three dissipation mechanisms

---

### Strategy B: GPT-5's Approach
**Method**: Coupling + Talagrand inequality
**Key Steps**:
1. Construct coupling between ρ_t^N and ρ_∞^N
2. Use Wasserstein contraction (doc-04, Theorem 4.1)
3. Apply Talagrand: D_KL ≤ C_T · W_2²
4. Chain inequalities to get KL-decay
5. Verify Talagrand constant C_T is N-uniform

**Strengths**:
- Reuses Wasserstein machinery from doc-04
- Probabilistic interpretation is intuitive

**Weaknesses**:
- Indirect path (W_2 → KL) may lose sharpness
- Talagrand constant C_T might depend on k (needs verification)
- Less explicit LSI constant

---

### Strategy Synthesis: Claude's Recommendation

**Chosen Method**: Lyapunov method (Gemini's approach)

**Rationale**:
- ✅ Direct proof gives explicit λ_LSI = f(σ², γ, λ_clone)
- ✅ Framework's synergistic dissipation lemma (verified: doc-03, Lemma 8.2) handles the hard part
- ✅ All constants proven N-uniform in framework
- ⚠ Trade-off: More technical Fisher information calculation, but framework provides bounds

**Integration**:
- Steps 1-2: Gemini's entropy production setup
- Step 3: Framework decomposition (doc-03, Theorem 8.1)
- Step 4-5: Gemini's dominance argument
- Critical insight from GPT-5: Verify k-independence explicitly (good sanity check)

**Verification Status**:
- ✅ All framework dependencies verified in glossary.md
- ✅ No circular reasoning detected
- ✅ All constants N-uniform as required

---

## III. Framework Dependencies

### Verified Dependencies

**Axioms**:
| Label | Statement | Used in Step | Verified |
|-------|-----------|--------------|----------|
| axiom-kinetic-dominance | σ² > σ_crit² = ... | Step 5 | ✅ |
| axiom-guaranteed-revival | κ_revival > 1 | QSD uniqueness | ✅ |

**Theorems**:
| Label | Document | Statement | Used in Step | Verified |
|-------|----------|-----------|--------------|----------|
| thm-synergistic-dissipation | doc-03 | Three-mechanism lower bound on Fisher | Step 4 | ✅ |
| thm-qsd-existence | doc-02 | Unique QSD for Euclidean Gas | Step 1 | ✅ |
| thm-cloning-keystone | doc-03 | Cloning contracts KL for fixed x | Step 3 | ✅ |

**Constants**:
| Symbol | Definition | Bound | Properties |
|--------|------------|-------|------------|
| λ_LSI | LSI constant | λ_LSI = σ² - σ_crit² | N-uniform, k-uniform |
| C_Fisher | Fisher lower bound | From synergistic dissipation | N-uniform |

---

## IV. Detailed Proof Sketch

### Overview

The proof establishes exponential KL-convergence by constructing a Lyapunov function
H(ρ) = D_KL(ρ || ρ_∞) and showing its time derivative satisfies a decay inequality.
The key innovation is the **synergistic dissipation** from three independent mechanisms:
velocity diffusion, cloning selection, and hypocoercive coupling. Under the Kinetic
Dominance Condition (σ² > σ_crit²), the dissipation overwhelms entropy production,
yielding exponential decay with an N-uniform rate.

### Proof Outline (Top-Level)

The proof proceeds in 6 main stages:

1. **Lyapunov Function Setup**: Define H(ρ_t) = D_KL(ρ_t || ρ_∞), verify properties
2. **Entropy Production Formula**: Compute dH/dt via generator
3. **Decomposition**: Split into kinetic (Ψ_kin) and cloning (Ψ_clone) contributions
4. **Synergistic Dissipation**: Apply framework's three-mechanism lower bound
5. **Dominance Argument**: Show dissipation > expansion under Kinetic Dominance
6. **LSI Conclusion**: Assemble Grönwall inequality, get exponential decay

---

### Step 1: Lyapunov Function Setup

**Goal**: Establish H(ρ_t) = D_KL(ρ_t^N || ρ_∞^N) as valid Lyapunov function

**Substep 1.1**: Verify ρ_∞ exists and is unique
- **Justification**: Framework Theorem (doc-02, thm-qsd-existence)
- **Why valid**: Axiom of Guaranteed Revival (axiom-revival) ensures uniqueness
- **Expected result**: ρ_∞ is unique invariant measure of P_Δt

**Substep 1.2**: Show H(ρ) ≥ 0 with equality iff ρ = ρ_∞
- **Justification**: Standard property of KL-divergence
- **Why valid**: Both ρ and ρ_∞ have support on full alive set 𝒜_N
- **Expected result**: H is proper Lyapunov (non-negative, zero at equilibrium)

**Substep 1.3**: Verify differentiability of H(ρ_t) in t
- **Justification**: Chain rule for KL along Markov evolution
- **Why valid**: Generator L exists (underdamped Langevin is smooth)
- **Expected result**: dH/dt = -I_Fisher(ρ_t) + A_jump(ρ_t)

**Dependencies**:
- Uses: {prf:ref}`thm-qsd-existence` (doc-02)
- Requires: {prf:ref}`axiom-guaranteed-revival` (doc-01)

**Potential Issues**: None (standard setup)

---

### Step 2: Entropy Production Formula

**Goal**: Derive dH/dt = ∫ (log(ρ/ρ_∞)) · L[ρ] dμ

**Substep 2.1**: Apply Fokker-Planck equation
- **Justification**: Evolution equation ∂_t ρ = L*[ρ] where L* is adjoint
- **Why valid**: Euclidean Gas generator is well-defined (doc-02, §3)
- **Expected result**: dH/dt = ∫ (log(ρ/ρ_∞)) · ∂_t ρ dμ

**Substep 2.2**: Integrate by parts to get Fisher-like form
- **Justification**: Integration by parts + boundary terms vanish
- **Why valid**: Compact support on alive set 𝒜_N
- **Expected result**: dH/dt = -I_Fisher(ρ) + boundary corrections

**Substep 2.3**: Identify boundary corrections as jump terms
- **Justification**: Cloning creates discontinuities in generator
- **Why valid**: Cloning operator is jump process
- **Expected result**: dH/dt = -I_Fisher + A_clone

**Dependencies**:
- Uses: Fokker-Planck equation (standard PDE)
- Requires: Compact support (framework guarantees via death boundary)

---

[... Steps 3-6 continue with same detailed structure ...]

---

### Step 6: LSI Conclusion

**Goal**: Assemble all parts to prove D_KL(ρ_t || ρ_∞) ≤ e^{-λ_LSI t} D_KL(ρ_0 || ρ_∞)

**Assembly**:
- From Step 2: dH/dt = -I_Fisher + A_clone
- From Step 4: I_Fisher ≥ C_Fisher · H (synergistic dissipation)
- From Step 5: |A_clone| ≤ ε_clone · H (cloning is contractive)
- Kinetic Dominance: C_Fisher > ε_clone

**Combining Results**:
$$
\frac{dH}{dt} \leq -(C_Fisher - \epsilon_clone) \cdot H = -\lambda_{LSI} \cdot H
$$

where λ_LSI := C_Fisher - ε_clone > 0 by Kinetic Dominance.

**Grönwall Inequality**:
Applying Grönwall's inequality to dH/dt ≤ -λ_LSI · H gives:
$$
H(t) \leq e^{-\lambda_{LSI} t} H(0)
$$

**Final Conclusion**:
Substituting H = D_KL:
$$
D_{KL}(\rho_t^N || \rho_\infty^N) \leq e^{-\lambda_{LSI} t} D_{KL}(\rho_0^N || \rho_\infty^N)
$$

**N-Uniformity Verification**:
- C_Fisher is N-uniform (doc-03, Theorem 8.1)
- ε_clone is N-uniform (doc-03, Lemma 7.3)
- Therefore λ_LSI = C_Fisher - ε_clone is N-uniform ✓

**Q.E.D.** ∎

---

## V. Technical Deep Dives

### Challenge 1: Synergistic Dissipation (Step 4)

**Why Difficult**: Need to show three independent dissipation mechanisms
(velocity diffusion, cloning selection, hypocoercive coupling) combine to give
tight Fisher information lower bound.

**Proposed Solution**:
Use framework's Theorem 8.1 (doc-03) which provides:
$$
I_{Fisher}(\rho) \geq C_\sigma \cdot I_v + C_\gamma \cdot I_{coupling} + C_\lambda \cdot I_clone
$$
where all constants C_* are N-uniform and positive.

**References**:
- Framework proof: doc-03, §8 (Synergistic Dissipation)
- Standard technique: Bakry-Émery calculus

---

[... Challenges 2-3 ...]

---

## VI. Proof Validation Checklist

- [x] **Logical Completeness**: All steps follow from previous
- [x] **Hypothesis Usage**: Kinetic Dominance used in Step 5
- [x] **Conclusion Derivation**: Exponential decay fully derived
- [x] **Framework Consistency**: All dependencies verified
- [x] **No Circular Reasoning**: No assumption of LSI in proof
- [x] **Constant Tracking**: λ_LSI = C_Fisher - ε_clone (N-uniform)
- [x] **Edge Cases**: k=1 handled via Axiom of Guaranteed Revival
- [x] **Regularity Verified**: Generator smoothness from doc-02

---

## VII. Alternative Approaches (Not Chosen)

### Alternative 1: Coupling + Talagrand

**Approach**: Construct coupling, bound W_2, apply Talagrand inequality

**Pros**:
- Reuses Wasserstein contraction machinery (doc-04)
- Probabilistic interpretation is intuitive

**Cons**:
- Talagrand constant may depend on k (needs verification)
- Indirect path loses sharpness in rate

**When to Consider**: If direct entropy production becomes too technical,
or if Talagrand constant can be shown k-independent.

---

## VIII. Open Questions

### Remaining Gaps
1. **Fisher Calculation Details**: Step 4 invokes framework's synergistic dissipation theorem - need to verify applicability of all preconditions
2. **Optimal λ_LSI**: Is λ_LSI = σ² - σ_crit² tight? Can it be improved?

### Extensions
1. **Time-Inhomogeneous Case**: Extend to non-constant σ(t), γ(t)
2. **Adaptive Gas**: Does N-uniform LSI persist with mean-field forces?

---

## IX. Expansion Roadmap

**Phase 1: Verify Framework Dependencies** (Estimated: 2 hours)
1. Read doc-03, §8 (Synergistic Dissipation) in full
2. Verify all preconditions of Theorem 8.1 are met
3. Check constants are as claimed (N-uniform)

**Phase 2: Fill Technical Details** (Estimated: 1 day)
1. Step 2: Expand entropy production derivation (integration by parts)
2. Step 4: Verify Fisher information decomposition
3. Step 5: Prove cloning expansion bound (may already be in doc-03)

**Phase 3: Add Rigor** (Estimated: 1 day)
1. Epsilon-delta arguments for continuity of H(t)
2. Measure-theoretic justification for Fokker-Planck
3. Edge case: k=1 (single alive walker)

**Phase 4: Review** (Estimated: 4 hours)
1. Framework cross-validation (all citations correct)
2. Constant tracking audit (ensure no hidden k-dependence)
3. Math Reviewer agent pass

**Total Estimated Expansion Time**: 3 days

---

## X. Cross-References

**Theorems Used**:
- {prf:ref}`thm-qsd-existence` (doc-02)
- {prf:ref}`thm-synergistic-dissipation` (doc-03)
- {prf:ref}`thm-cloning-keystone` (doc-03)

**Definitions Used**:
- {prf:ref}`def-kl-divergence` (doc-01)
- {prf:ref}`def-fisher-information` (doc-09)

---

✅ Proof sketch written to: docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

---

## Customization Options

### Depth Levels

**Quick** (~15 min):
```
Depth: quick
```
Sketches main theorem only, high-level strategy

**Thorough** (~45 min, DEFAULT):
```
Depth: thorough
```
Sketches main theorem + key supporting lemmas

**Exhaustive** (~2 hours):
```
Depth: exhaustive
```
Sketches all theorems and lemmas in document

### Focus on Specific Theorems

**By label**:
```
Theorems: thm-main-result, lemma-key-bound, lemma-technical-step
```

**By topic**:
```
Focus on: LSI convergence proof, N-uniform constants, Keystone Principle
```

**By section**:
```
Focus on: Section 5 (main theorem), Appendix A (combinatorial lemma)
```

---

## Real Example

I'll demonstrate with an actual theorem:

```
Load proof-sketcher agent from .claude/agents/proof-sketcher.md

Sketch proof for:
Document: docs/source/1_euclidean_gas/04_wasserstein_contraction.md
Theorem: thm-wasserstein-contraction

Depth: thorough

Focus on: Coupling construction and contraction rate derivation
```

**What the agent will do**:
1. Extract theorem statement (lines ~200-220)
2. Identify dependencies (cloning operator, kinetic operator)
3. Submit to Gemini 2.5 Pro + GPT-5 Pro with identical prompts
4. Gemini suggests: "Use maximal coupling + drift analysis"
5. GPT-5 suggests: "Construct coupling via Wasserstein interpolation"
6. Agent compares both, verifies framework dependencies
7. Synthesizes: "Use Gemini's maximal coupling (simpler), GPT-5's drift bound calculation"
8. Writes complete sketch to `sketcher/sketch_{timestamp}_proof_04_wasserstein_contraction.md`

**Time**: ~40 minutes

---

## Parallel Execution Example

Sketch proofs for 3 theorems simultaneously:

```
Launch 3 proof-sketcher agents in parallel:

Agent 1: Sketch docs/source/1_euclidean_gas/04_wasserstein_contraction.md
         Theorem: thm-wasserstein-contraction
         Depth: thorough

Agent 2: Sketch docs/source/1_euclidean_gas/06_convergence.md
         Theorem: thm-fl-euclidean
         Depth: thorough

Agent 3: Sketch docs/source/1_euclidean_gas/09_kl_convergence.md
         Theorem: thm-kl-convergence-euclidean
         Depth: thorough

Provide 3 separate proof sketches.
```

All 3 will run independently and complete around the same time (~45 min each).

---

## Comparison: Agent vs Manual Proof Planning

| Feature | Manual Planning | Proof Sketcher Agent |
|---------|----------------|----------------------|
| **Time** | 2-4 hours | ~45 min |
| **Coverage** | May miss dependencies | All dependencies verified |
| **Alternatives** | Usually stick to first idea | Documents 2+ approaches |
| **Framework Check** | Manual cross-referencing | Automatic glossary lookup |
| **Rigor** | Varies by expertise | Consistent checklist |
| **Documentation** | Often informal notes | Structured 10-section format |
| **Expansion Ready** | Needs reorganization | Immediately expandable |

---

## Tips

1. **Start with thorough depth** - it's the sweet spot for most theorems

2. **Be specific about focus** - agent will prioritize those aspects:
   ```
   Focus on: Verify all constants are N-uniform, track k-dependence
   ```

3. **Check file path first**:
   ```
   ls -lh docs/source/1_euclidean_gas/09_kl_convergence.md
   ```

4. **For huge documents** (>3000 lines), focus on specific theorems:
   ```
   Theorem: thm-main-result-only
   ```
   (Don't ask for exhaustive depth)

5. **After sketching**, read Section IX (Expansion Roadmap) for next steps

6. **Use multiple passes** for complex theorems:
   - Pass 1: Get overall strategy
   - Prove missing lemmas
   - Pass 2: Re-sketch with lemmas available

---

## Next Steps After Sketching

1. **Read Section II** (Strategy Comparison) - understand why this approach was chosen
2. **Read Section III** (Framework Dependencies) - verify you have all ingredients
3. **Read Section IV** (Detailed Sketch) - understand step-by-step plan
4. **Read Section IX** (Expansion Roadmap) - follow the implementation plan
5. **Expand sketch to full proof** (manually or with assistance)
6. **Quality control with Math Reviewer** agent

---

That's it! Just copy-paste the simple usage above to get started.

For more details, see:
- Full agent definition: `.claude/agents/proof-sketcher.md`
- Complete docs: `.claude/agents/proof-sketcher-README.md`
- Framework context: `CLAUDE.md`
