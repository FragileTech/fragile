# Theorem Prover Agent - Documentation

**Version**: 1.0
**Created**: 2025-10-24
**Purpose**: Expand proof sketches into complete, publication-ready proofs at Annals of Mathematics standard

---

## Overview

The **Theorem Prover** is an autonomous agent that takes proof sketches (from Proof Sketcher) and expands them into complete, rigorous proofs suitable for top-tier mathematics journals by comparing expansions from Gemini 2.5 Pro and GPT-5 Pro.

### Key Position in Workflow

```
Complete Proof Development Pipeline:

1. Proof Sketcher → Strategy outline (sketcher/sketch_*.md)
2. [Prove missing lemmas if needed]
3. **Theorem Prover → Complete proof (proofs/proof_*.md)** ← THIS AGENT
4. Math Reviewer → Quality control (reviewer/review_*.md)
5. Fix issues → Publication ready ✓
```

### Core Distinction

| Feature | Proof Sketcher | Theorem Prover | Math Reviewer |
|---------|----------------|----------------|---------------|
| **Input** | Theorem statement | Proof sketch | Complete proof |
| **Output** | Strategy (3-7 steps) | Full proof | Error report |
| **Goal** | Plan proof | Execute proof | Find errors |
| **Depth** | High-level steps | Every epsilon | Verify correctness |
| **Length** | ~100-200 lines | ~500-2000 lines | ~50-100 issues |
| **Time** | ~45 min | ~2-4 hours | ~45 min |
| **When** | Before proving | While proving | After proving |

---

## Core Capabilities

### Proof Expansion Generation
- Submits identical expansion prompts to Gemini 2.5 Pro (structure) + GPT-5 Pro (calculations)
- Expands each proof step to complete rigor
- Fills all epsilon-delta arguments
- Justifies all measure-theoretic operations
- Handles all edge cases explicitly

### Rigor Standards (Annals of Mathematics)
- ✅ **Epsilon-delta**: All limits proven (no "clearly approaches")
- ✅ **Measure theory**: All Fubini/DCT conditions verified
- ✅ **Constants**: All have explicit formulas (no unjustified O(1))
- ✅ **Edge cases**: k=1, N→∞, boundary all handled
- ✅ **Counterexamples**: For necessity of hypotheses
- ✅ **No handwaving**: Every "obviously", "trivially" expanded

### Framework Verification
- Cross-validates all dependencies against `docs/glossary.md`
- Ensures no circular reasoning (proof doesn't assume conclusion)
- Tracks all constants with explicit bounds
- Verifies all N-uniformity and k-uniformity claims

### Technical Analysis
- Compares rigor of both expansions (Gemini vs GPT-5)
- Synthesizes optimal proof from best elements
- Identifies mathematical contradictions
- Provides publication readiness assessment

---

## How to Use

### Method 1: From Sketch File

```
Load theorem-prover agent.

Expand proof sketch:
docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

### Method 2: Direct Theorem (Auto-Find Sketch)

```
Load theorem-prover.

Expand proof for theorem: thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

Agent will search `sketcher/` for most recent sketch matching this theorem.

### Method 3: With Focus Areas

```
Load theorem-prover.

Expand proof sketch: sketcher/sketch_20251024_1530_proof_*.md
Focus on: Step 4 (synergistic dissipation) - add complete Fisher information derivation
```

### Method 4: Expand Specific Steps Only

```
Expand steps 3-5 from sketch:
sketcher/sketch_20251024_1530_proof_09_kl_convergence.md

Keep steps 1-2, 6-7 as sketched (will expand later)
```

---

## Input Parameters

### Required
- **sketch_path**: Path to proof sketch file
  - Example: `docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md`

OR

- **theorem_label**: Theorem label (agent finds sketch)
  - Example: `thm-kl-convergence-euclidean`
- **document**: Document containing theorem
  - Example: `docs/source/1_euclidean_gas/09_kl_convergence.md`

### Optional
- **focus_steps**: Specific steps to expand in full detail
  - Example: `Step 4, Step 5`
  - Others kept at sketch level

- **depth**: Rigor level
  - `standard`: Normal Annals rigor (default)
  - `maximum`: Every epsilon, every detail

---

## Output Format

### File Location
```
docs/source/1_euclidean_gas/proofs/proof_20251024_1630_thm_kl_convergence_euclidean.md
```

**Pattern**: `proofs/proof_{YYYYMMDD_HHMM}_{theorem_label}.md`

### Proof Structure (9 sections)

1. **Theorem Statement** (exact from sketch)
2. **Proof Expansion Comparison**
   - Gemini's version with rigor assessment
   - GPT-5's version with rigor assessment
   - Claude's synthesis with rationale
3. **Framework Dependencies** (verified tables)
   - Axioms used
   - Theorems used
   - Definitions used
   - Constants tracked
4. **Complete Rigorous Proof** (main content)
   - Every step expanded to full detail
   - All epsilon-delta arguments
   - All measure theory justified
   - All edge cases handled
5. **Verification Checklist**
   - Logical rigor ✓
   - Measure theory ✓
   - Constants ✓
   - Edge cases ✓
   - Framework consistency ✓
6. **Edge Cases and Special Situations**
   - k=1 (single walker)
   - N→∞ (thermodynamic limit)
   - Boundary conditions
   - Degenerate cases
7. **Counterexamples for Necessity**
   - For each hypothesis
   - Shows hypothesis can't be weakened
8. **Publication Readiness Assessment**
   - Rigor scores (1-10)
   - Annals of Mathematics verdict
   - Remaining tasks if any
9. **Cross-References**
   - All theorems cited
   - All lemmas cited
   - All definitions used

---

## Example Output Preview

```markdown
# Complete Proof for thm-kl-convergence-euclidean

**Source Sketch**: sketcher/sketch_20251024_1530_proof_09_kl_convergence.md

## II. Proof Expansion Comparison

### Expansion A: Gemini's Version
**Rigor Level**: 8/10
- Epsilon-delta: Mostly complete, one gap in Substep 4.2
- Measure theory: All Fubini conditions verified ✓
- Constants: Most explicit, C_3 only stated as O(1)
- Edge cases: k=1 and boundary both handled ✓

**Key Strengths**:
- Clean overall structure
- Excellent measure-theoretic justifications
- Thorough edge case analysis

**Key Weaknesses**:
- Substep 4.2 limit not fully proven (missing ε-δ)
- Constant C_3 not given explicit formula

### Expansion B: GPT-5's Version
**Rigor Level**: 9/10
- Epsilon-delta: All limits fully proven ✓
- Measure theory: Fubini applied but condition 2 not verified
- Constants: All have explicit formulas ✓
- Edge cases: k=1 handled, boundary only mentioned

**Key Strengths**:
- Complete epsilon-delta arguments
- All constants have explicit formulas
- Detailed bound calculations

**Key Weaknesses**:
- Fubini condition 2 asserted without verification
- Boundary case not fully analyzed

### Synthesis: Claude's Complete Proof

**Chosen Elements**:
| Component | Source | Reason |
|-----------|--------|--------|
| Structure | Gemini | Cleaner logical flow |
| Step 1-2 | Gemini | Already complete |
| Step 3 | GPT-5 | More detailed calculation |
| Step 4 | GPT-5 + Gemini | GPT-5's ε-δ + Gemini's Fubini |
| Step 5-6 | Gemini | Already rigorous |
| Constants | GPT-5 | All explicit formulas |
| Edge cases | Gemini | Both k=1 and boundary |

**Quality**: ✅ Meets Annals of Mathematics standard

## IV. Complete Rigorous Proof

:::{prf:proof}

We prove the theorem in 6 main steps.

### Step 1: Define Entropy Lyapunov Function

**Goal**: Establish H(ρ_t) = D_KL(ρ_t || ρ_∞) as valid Lyapunov

Let ρ_∞ denote the unique QSD (quasi-stationary distribution) of the Euclidean Gas Markov operator P_Δt. By {prf:ref}`thm-qsd-existence` (document 02_euclidean_gas, Theorem 4.3), such a ρ_∞ exists and is unique under the Axiom of Guaranteed Revival ({prf:ref}`axiom-guaranteed-revival`, document 01_fragile_gas_framework).

**Verification of preconditions**:
- Precondition 1: P_Δt is time-homogeneous ✓ (by definition of Euclidean Gas)
- Precondition 2: Axiom of Guaranteed Revival holds ✓ (κ_revival > 1 by framework assumption)

Define the Kullback-Leibler divergence:
$$
H(ρ_t) := D_{KL}(\rho_t^N || \rho_\infty^N) = \int_{\Sigma_N} \rho_t^N \log\left(\frac{\rho_t^N}{\rho_\infty^N}\right) d\mu
$$

**Property 1** (Non-negativity): H(ρ_t) ≥ 0 for all ρ_t.

*Proof*: By Gibbs' inequality, D_KL(ρ || ν) ≥ 0 with equality if and only if ρ = ν μ-almost everywhere. ∎

**Property 2** (Zero at equilibrium): H(ρ_∞) = 0.

*Proof*: Direct from definition: D_KL(ρ_∞ || ρ_∞) = ∫ ρ_∞ · log(1) dμ = 0. ∎

**Property 3** (Differentiability): The map t ↦ H(ρ_t) is differentiable in t with:
$$
\frac{dH}{dt} = \int_{\Sigma_N} \left(\log\frac{\rho_t}{\rho_\infty} + 1\right) \frac{\partial \rho_t}{\partial t} d\mu
$$

*Proof*: By Fokker-Planck equation, ∂ρ_t/∂t = L*[ρ_t] where L* is the adjoint generator. The integral is well-defined because ρ_t has compact support on alive set 𝒜_N (dead walkers have finite mass), and the integrand is bounded. Differentiating under the integral sign (justified by dominated convergence with dominating function g(x) = 2ρ_∞ + 2ρ_t, which is integrable) gives the stated formula. ∎

**Conclusion of Step 1**: We have rigorously established H(ρ_t) = D_KL(ρ_t || ρ_∞) as a valid Lyapunov function satisfying:
- H ≥ 0 (non-negative)
- H = 0 ⟺ ρ = ρ_∞ (zero at equilibrium)
- H is differentiable in t

This will be used in Step 2 to compute the entropy production.

---

[... Steps 2-6 continue with same level of detail ...]

---

### Step 6: Assembly and Conclusion

From Step 2, we have the entropy production formula:
$$
\frac{dH}{dt} = -I_{\text{Fisher}}(\rho_t) + A_{\text{clone}}(\rho_t)
$$

From Step 4, synergistic dissipation gives:
$$
I_{\text{Fisher}}(\rho_t) \geq C_{\text{Fisher}} \cdot H(\rho_t)
$$
where C_Fisher = σ²(2γ - γ²τ/2) - ε_v (explicit formula from {prf:ref}`thm-synergistic-dissipation`).

From Step 5, cloning expansion is bounded:
$$
|A_{\text{clone}}(\rho_t)| \leq \epsilon_{\text{clone}} \cdot H(\rho_t)
$$
where ε_clone = λ_max · (1 - exp(-λ_clone·Δt)) (from {prf:ref}`lemma-cloning-expansion`).

**Kinetic Dominance** (hypothesis of theorem): σ² > σ_crit² implies C_Fisher > ε_clone.

Define λ_LSI := C_Fisher - ε_clone > 0.

Combining all results:
$$
\frac{dH}{dt} = -I_{\text{Fisher}} + A_{\text{clone}} \leq -C_{\text{Fisher}} \cdot H + \epsilon_{\text{clone}} \cdot H = -(C_{\text{Fisher}} - \epsilon_{\text{clone}}) \cdot H = -\lambda_{\text{LSI}} \cdot H
$$

**Grönwall's inequality**: The differential inequality dH/dt ≤ -λ_LSI · H with H(0) = D_KL(ρ_0 || ρ_∞) has solution:
$$
H(t) \leq e^{-\lambda_{\text{LSI}} t} H(0)
$$

Substituting H = D_KL:
$$
D_{KL}(\rho_t^N || \rho_\infty^N) \leq e^{-\lambda_{\text{LSI}} t} D_{KL}(\rho_0^N || \rho_\infty^N)
$$

**N-uniformity verification**:
- C_Fisher is N-uniform: from {prf:ref}`thm-synergistic-dissipation`, explicitly verified
- ε_clone is N-uniform: from {prf:ref}`lemma-cloning-expansion`, explicitly verified
- Therefore λ_LSI = C_Fisher - ε_clone is N-uniform ✓

This is precisely the statement of the theorem. Q.E.D. ∎

:::

## VIII. Publication Readiness Assessment

**Mathematical Rigor**: 9/10
- All epsilon-delta complete
- All measure theory justified
- One minor notation inconsistency (fixed in synthesis)

**Completeness**: 9/10
- All claims justified
- All edge cases handled
- Counterexamples for all hypotheses

**Clarity**: 8/10
- Logical flow excellent
- Some technical details dense (unavoidable for rigor)

**Framework Consistency**: 10/10
- All dependencies verified
- All constants explicit
- No circular reasoning

**Overall Assessment**: **MEETS ANNALS OF MATHEMATICS STANDARD**

✅ Ready for submission after minor polish (notation consistency check)
```

---

## Workflow

### What Happens When You Invoke

1. **Phase 1: Sketch Analysis** (~15 min)
   - Agent reads proof sketch file
   - Extracts theorem, strategy, steps, dependencies
   - Identifies expansion requirements
   - Checks for missing lemmas

2. **Phase 2: Prompt Preparation** (~10 min)
   - Constructs detailed expansion prompt for EACH step
   - Includes all framework dependencies
   - Specifies rigor requirements (epsilon-delta, measure theory, etc.)

3. **Phase 3: Dual Expansion** (~60-120 min)
   - Submits EACH step to Gemini 2.5 Pro + GPT-5 Pro in parallel
   - Waits for both expansions to complete
   - Parses both outputs for comparison

4. **Phase 4: Critical Comparison** (~40-60 min)
   - Scores rigor of both expansions (13-point checklist)
   - Identifies contradictions and resolves
   - Verifies all framework dependencies
   - Synthesizes optimal complete proof

5. **Phase 5: Proof Generation** (~15-20 min)
   - Formats complete proof (9 sections)
   - Writes to `proofs/proof_{timestamp}_{theorem_label}.md`
   - Reports publication readiness assessment

**Total Time**:
- Standard proof (3-5 steps): ~2-3 hours
- Complex proof (6-8 steps): ~3-4 hours
- Very complex (9+ steps): ~5-6 hours

---

## Best Practices

### When to Use Theorem Prover

✅ **Use for**:
- Expanding proof sketches to complete proofs
- Preparing proofs for publication submission
- Ensuring top-tier journal rigor
- Filling all epsilon-delta and measure theory details
- Generating complete proofs from strategies

❌ **Don't use for**:
- Creating initial proof strategies (use Proof Sketcher instead)
- Finding errors in existing proofs (use Math Reviewer instead)
- Trivial theorems with obvious proofs (just write directly)
- Theorems without sketches (run Proof Sketcher first)

### Tips for Best Results

1. **Always sketch first**:
   ```
   Step 1: Proof Sketcher (create strategy)
   Step 2: Prove missing lemmas if needed
   Step 3: Theorem Prover (expand to full proof)
   ```

2. **Handle missing dependencies**:
   - If sketch says "Requires Lemma X (not yet proven)"
   - Sketch and prove Lemma X first
   - Then expand main theorem

3. **Focus on hard steps**:
   ```
   Expand steps 4-5 from sketch (these are the technical challenges)
   Keep steps 1-3, 6-7 as sketched for now
   ```

4. **Iterate if needed**:
   - First expansion: Standard rigor
   - If assessment says "needs polish"
   - Second expansion: Maximum rigor on flagged steps

5. **Use Math Reviewer after**:
   ```
   Step 1: Theorem Prover → complete proof
   Step 2: Math Reviewer → find any remaining gaps
   Step 3: Fix gaps → Publication ready
   ```

---

## Agent Guarantees

The Theorem Prover agent guarantees:

1. ✅ **Complete Rigor**: All epsilon-delta, all measure theory, all edge cases
2. ✅ **Explicit Constants**: No unjustified O(1) or "sufficiently small" without bound
3. ✅ **Framework Consistency**: All dependencies verified against glossary.md
4. ✅ **No Circular Reasoning**: Proof doesn't assume conclusion
5. ✅ **Counterexamples**: For necessity of all hypotheses
6. ✅ **Publication Standard**: Suitable for Annals of Mathematics
7. ✅ **Dual Validation**: Gemini + GPT-5 cross-check for errors

---

## Troubleshooting

### Problem: Missing Lemmas Detected

**Symptoms**: Agent says "MISSING DEPENDENCIES DETECTED"

**Cause**: Proof sketch identified lemmas that aren't proven yet

**Solutions**:
1. **Option A** (recommended): Let agent sketch and prove lemmas first
   ```
   Choose option 1: Sketch and prove lemmas first
   Agent will recursively handle dependencies
   ```

2. **Option B**: You provide proofs
   ```
   Write proofs of missing lemmas manually
   Then re-run Theorem Prover
   ```

3. **Option C**: Mark proof as CONDITIONAL
   ```
   Agent proceeds assuming lemmas
   Proof marked as incomplete
   ```

### Problem: Low Rigor Assessment

**Symptoms**: Assessment says "MAJOR REVISION NEEDED", scores < 7/10

**Cause**: Expansions have gaps in epsilon-delta or measure theory

**Solutions**:
1. Check which steps have gaps (agent lists them)
2. Re-expand with focus on problematic steps:
   ```
   Expand steps 4, 7 from sketch with maximum rigor
   ```
3. Or revise proof strategy (may need different approach)

### Problem: Contradictory Expansions

**Symptoms**: Agent reports "MATHEMATICAL CONTRADICTION"

**Cause**: Gemini and GPT-5 give different results for same step

**What Agent Does**:
- Analyzes both
- Verifies against framework
- Determines which is correct
- Uses correct version in synthesis

**User Action**: Review agent's resolution, verify if uncertain

### Problem: Expansion Takes Too Long

**Symptoms**: Agent running >6 hours for single theorem

**Causes**:
- Proof sketch has many steps (9+)
- Steps are very technical
- Both AIs proposing complex expansions

**Solutions**:
- Expand in stages (steps 1-3, then 4-6, then 7-9)
- Use `depth: standard` instead of `maximum`
- Try again later if MCP services are slow

---

## Advanced Usage

### Incremental Expansion

For very complex proofs, expand incrementally:

```
Session 1: Expand steps 1-3
  Produces: proof_partial_step1_3_{theorem}.md

Session 2: Expand steps 4-6 using step 1-3 results
  Produces: proof_partial_step4_6_{theorem}.md

Session 3: Combine and expand remaining
  Produces: proof_complete_{theorem}.md
```

### Parallel Expansion of Related Theorems

```
Run 2 theorem-prover agents in parallel:

Agent 1: Expand proof for thm-main-result
         Sketch: sketcher/sketch_*_thm_main.md

Agent 2: Expand proof for lemma-key-technical
         Sketch: sketcher/sketch_*_lemma_key.md

Both produce complete proofs simultaneously (~3 hours each)
```

### Handling Lemma Dependencies

Automatic recursive expansion:

```
Load theorem-prover.

Expand proof for: thm-complex-result
Document: docs/source/1_euclidean_gas/09_kl_convergence.md

When prompted about missing lemmas: Choose option 1 (sketch and prove first)

Agent will:
1. Detect lemma-A, lemma-B are missing
2. Run Proof Sketcher on lemma-A
3. Run Theorem Prover on lemma-A
4. Run Proof Sketcher on lemma-B
5. Run Theorem Prover on lemma-B
6. Finally expand main theorem using proven lemmas
```

---

## Model Configuration

### Pinned Models (DO NOT CHANGE unless explicitly instructed):

- **Gemini**: `gemini-2.5-pro`
  - Strength: Proof structure, abstract reasoning, measure theory
  - Use case: Overall proof architecture, rigorous justifications

- **GPT-5**: `gpt-5` with `model_reasoning_effort=high`
  - Strength: Detailed calculations, epsilon-delta, bound derivations with deep reasoning
  - Use case: Technical expansions, explicit formulas

### Why Two Models?

Different AI models excel at different aspects of rigor:
- **Gemini**: Better at high-level structure, abstract arguments, measure theory
- **GPT-5**: Better at detailed calculations, epsilon-delta arguments, constant tracking
- **Synthesis**: Combining both provides most rigorous complete proof

---

## Output File Management

### File Naming Convention
```
proof_{YYYYMMDD_HHMM}_{theorem_label}.md
```

**Examples**:
- `proof_20251024_1630_thm_kl_convergence_euclidean.md`
- `proof_20251024_1645_lemma_synergistic_dissipation.md`

### Directory Structure
```
docs/source/1_euclidean_gas/
├── 09_kl_convergence.md                          # Original document
├── sketcher/                                      # Proof sketches
│   └── sketch_20251024_1530_proof_09_kl_convergence.md
├── proofs/                                        # Complete proofs (NEW)
│   └── proof_20251024_1630_thm_kl_convergence_euclidean.md
└── reviewer/                                      # Quality control
    └── review_20251024_1700_proof_20251024_1630_thm_kl.md
```

### Version Management

Multiple proofs for same theorem (iterative refinement):
- Timestamp prevents overwriting
- Compare versions to see improvements
- Keep history of expansion attempts

**Example workflow**:
```
proof_20251024_1630_thm_X.md  → First expansion (rigor 7/10)
proof_20251024_1800_thm_X.md  → Re-expanded hard steps (rigor 9/10)
proof_20251024_1900_thm_X.md  → Final version after review (rigor 10/10)
```

---

## Integration with Workflow

### Recommended Proof Development Workflow

```
1. Initial Strategy → Proof Sketcher (thorough)
   Output: sketcher/sketch_*.md
   ↓
2. Review sketch → Identify missing lemmas
   ↓
3. Prove missing lemmas → Proof Sketcher + Theorem Prover for each
   Output: proofs/proof_*_lemma_*.md
   ↓
4. Expand main theorem → Theorem Prover
   Output: proofs/proof_*_thm_*.md
   ↓
5. Quality control → Math Reviewer (exhaustive)
   Output: reviewer/review_*_proof_*.md
   ↓
6. Fix issues identified → Edit proofs/proof_*.md
   ↓
7. Final validation → Math Reviewer (thorough)
   ↓
8. Publication ready ✓
```

### Collaboration with Other Tools

- **Before Theorem Prover**: Proof Sketcher creates strategy
- **After Theorem Prover**: Math Reviewer validates rigor
- **Parallel with Theorem Prover**: Can expand multiple theorems simultaneously
- **Iterative**: Can re-expand with feedback from Math Reviewer

---

## Limitations

### What Theorem Prover CANNOT Do

❌ **Create proof strategies**: Only expands existing sketches (use Proof Sketcher)
❌ **Find errors**: Only expands proofs (use Math Reviewer for validation)
❌ **Prove theorems automatically**: Requires human-created sketch as input
❌ **Guarantee mathematical correctness**: Expansions need review
❌ **Handle theorems outside framework**: Requires Fragile framework context

### What Theorem Prover CAN Do

✅ **Expand sketches to full rigor**: Every epsilon-delta, every measure theory detail
✅ **Track all constants**: Explicit formulas for every bound
✅ **Handle all edge cases**: k=1, N→∞, boundary, degenerate
✅ **Verify framework consistency**: Cross-check all dependencies
✅ **Provide counterexamples**: Show necessity of hypotheses
✅ **Assess publication readiness**: Honest evaluation against top-tier standards

---

## FAQs

**Q: How is this different from Proof Sketcher?**

A: Proof Sketcher creates the strategy (3-7 high-level steps). Theorem Prover expands that strategy to full detail (every epsilon-delta, every edge case). Think: architect (Sketcher) vs builder (Prover).

**Q: Can I expand a theorem without sketching first?**

A: Not recommended. The sketch provides the verified strategy and framework dependencies. Without it, Theorem Prover has no guidance on proof approach.

**Q: What if the expansion still has gaps?**

A: Run Math Reviewer on the expanded proof. It will identify remaining gaps. Fix those, then re-run Theorem Prover if needed.

**Q: Can I use custom models instead of Gemini/GPT-5?**

A: Models are pinned for consistency. You can override:
```
Use claude-opus-4 instead of gemini for proof expansion
```
But not recommended (agent optimized for Gemini/GPT-5).

**Q: How do I know if a proof is "publication ready"?**

A: Check Section VIII (Publication Readiness Assessment). If it says "MEETS STANDARD" with scores ≥8/10, it's ready. If lower, address the listed remaining tasks.

**Q: What if both expansions have errors?**

A: Agent identifies contradictions and resolves by verifying against framework. If both are wrong, agent provides correct version. If unsure, flags for user decision.

---

## Support

For issues or questions:
1. Check this README
2. See QUICKSTART guide for examples
3. Consult `theorem-prover.md` for agent internals
4. Check CLAUDE.md § Mathematical Proofing
5. Open issue: https://github.com/anthropics/claude-code/issues

---

**Version**: 1.0
**Last Updated**: 2025-10-24
**Maintainer**: Fragile Framework Team
