# Theorem Prover Agent - Quick Start Guide

## Simplest Usage (Copy-Paste Ready)

### Expand from Sketch File

Just paste this into Claude:

```
Load the theorem-prover agent from .claude/agents/theorem-prover.md
and expand proof sketch:

docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

### Expand by Theorem Label (Auto-Find Sketch)

```
Load theorem-prover agent.

Expand proof for theorem: thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

Agent will find the most recent sketch for this theorem automatically.

---

## What Happens

1. **Agent reads proof sketch** from `sketcher/` folder
2. **Extracts strategy**: Chosen approach, steps, dependencies from Section II & IV
3. **Identifies requirements**: Which steps need epsilon-delta, measure theory, edge cases
4. **Checks dependencies**: Flags any missing lemmas
5. **Prepares expansion prompts** for each step (identical for both AIs)
6. **Submits to Gemini 2.5 Pro + GPT-5 Pro in parallel** for EACH step
7. **Waits for both expansions** to complete all steps
8. **Critically compares** expansions:
   - Scores rigor (13-point checklist per step)
   - Identifies contradictions
   - Verifies framework dependencies
9. **Synthesizes optimal proof** combining best elements
10. **Generates complete proof** with 9 sections:
    - Theorem statement
    - Expansion comparison (Gemini vs GPT-5 vs synthesis)
    - Framework dependencies (verified tables)
    - Complete rigorous proof (every detail)
    - Verification checklist
    - Edge cases handled
    - Counterexamples for necessity
    - Publication readiness assessment
    - Cross-references
11. **Writes to file**: `proofs/proof_{timestamp}_{theorem_label}.md`

---

## Expected Output Format

You'll receive a complete proof like this:

```markdown
# Complete Proof for thm-kl-convergence-euclidean

**Source Sketch**: sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
**Generated**: 2025-10-24 16:30
**Agent**: Theorem Prover v1.0

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

## II. Proof Expansion Comparison

### Expansion A: Gemini's Version
**Rigor Level**: 8/10 - Mostly complete, one epsilon-delta gap in Step 4

**Completeness**:
- Epsilon-delta arguments: Mostly complete (gap in Substep 4.2)
- Measure theory: All verified ✓
- Constant tracking: Most explicit (C_3 only stated as O(1))
- Edge cases: All handled (k=1, boundary both covered) ✓

**Key Strengths**:
1. Excellent measure-theoretic justifications (all Fubini conditions verified)
2. Thorough edge case analysis
3. Clean overall structure

**Key Weaknesses**:
1. Substep 4.2 limit not fully epsilon-delta proven
2. Constant C_3 not given explicit formula

---

### Expansion B: GPT-5's Version
**Rigor Level**: 9/10 - Nearly complete, one measure theory gap

**Completeness**:
- Epsilon-delta arguments: All complete ✓
- Measure theory: Mostly verified (Fubini condition 2 asserted)
- Constant tracking: All explicit ✓
- Edge cases: k=1 handled, boundary only mentioned

**Key Strengths**:
1. Complete epsilon-delta arguments (all limits fully proven)
2. All constants have explicit formulas
3. Detailed bound calculations

**Key Weaknesses**:
1. Fubini condition 2 not explicitly verified
2. Boundary case analysis incomplete

---

### Synthesis: Claude's Complete Proof

**Chosen Elements**:
| Component | Source | Reason |
|-----------|--------|--------|
| Structure | Gemini | Cleaner logical flow |
| Step 1-2 | Gemini | Already fully rigorous |
| Step 3 | GPT-5 | More detailed calculation |
| Step 4 | GPT-5 + Gemini | GPT-5's ε-δ + Gemini's Fubini |
| Step 5-6 | Gemini | Already rigorous |
| Constants | GPT-5 | All have explicit formulas |
| Edge cases | Gemini | Both k=1 and boundary |

**Quality**: ✅ Meets Annals of Mathematics standard

---

## IV. Complete Rigorous Proof

:::{prf:proof}

We prove the theorem in 6 main steps.

---

### Step 1: Define Entropy Lyapunov Function

**Goal**: Establish H(ρ_t) = D_KL(ρ_t || ρ_∞) as valid Lyapunov function

Let ρ_∞ denote the unique QSD of the Euclidean Gas Markov operator P_Δt. By {prf:ref}`thm-qsd-existence` (document 02_euclidean_gas, Theorem 4.3), such a ρ_∞ exists and is unique under the Axiom of Guaranteed Revival.

**Verification of preconditions**:
- Precondition 1: P_Δt is time-homogeneous ✓ (by definition)
- Precondition 2: Axiom of Guaranteed Revival holds ✓ (κ_revival > 1)

Define:
$$
H(\rho_t) := D_{KL}(\rho_t^N || \rho_\infty^N) = \int_{\Sigma_N} \rho_t^N \log\left(\frac{\rho_t^N}{\rho_\infty^N}\right) d\mu
$$

**Property 1** (Non-negativity): H(ρ_t) ≥ 0 for all ρ_t.

*Proof*: By Gibbs' inequality, D_KL(ρ || ν) ≥ 0 with equality iff ρ = ν μ-a.e. ∎

**Property 2** (Zero at equilibrium): H(ρ_∞) = 0.

*Proof*: D_KL(ρ_∞ || ρ_∞) = ∫ ρ_∞ · log(1) dμ = 0. ∎

**Property 3** (Differentiability): t ↦ H(ρ_t) is differentiable with:
$$
\frac{dH}{dt} = \int_{\Sigma_N} \left(\log\frac{\rho_t}{\rho_\infty} + 1\right) \frac{\partial \rho_t}{\partial t} d\mu
$$

*Proof*: By Fokker-Planck equation, ∂ρ_t/∂t = L*[ρ_t]. Differentiating under integral (justified by dominated convergence with dominating function g = 2ρ_∞ + 2ρ_t ∈ L¹) gives the formula. ∎

**Edge case k=1**: When k=1, the QSD collapses to delta measure at single walker position. Lyapunov function still well-defined: H(δ_x || δ_x) = 0. ✓

**Conclusion**: H is valid Lyapunov (non-negative, zero at equilibrium, differentiable).

---

### Step 2: Compute Entropy Production

{... full rigorous expansion continues ...}

---

[... Steps 3-6 with same level of detail ...]

---

### Step 6: Assembly and Conclusion

From Steps 2-5:
- Step 2: dH/dt = -I_Fisher + A_clone
- Step 4: I_Fisher ≥ C_Fisher · H where C_Fisher = σ²(2γ - γ²τ/2) - ε_v
- Step 5: |A_clone| ≤ ε_clone · H where ε_clone = λ_max(1 - e^(-λ·Δt))

Kinetic Dominance hypothesis: σ² > σ_crit² implies C_Fisher > ε_clone.

Define λ_LSI := C_Fisher - ε_clone > 0.

Combining:
$$
\frac{dH}{dt} \leq -C_{\text{Fisher}} H + \epsilon_{\text{clone}} H = -\lambda_{\text{LSI}} H
$$

By Grönwall: H(t) ≤ e^(-λ_LSI t) H(0).

Substituting H = D_KL:
$$
D_{KL}(\rho_t^N || \rho_\infty^N) \leq e^{-\lambda_{LSI} t} D_{KL}(\rho_0^N || \rho_\infty^N)
$$

**N-uniformity**: C_Fisher and ε_clone both N-uniform (verified in framework) → λ_LSI N-uniform ✓

Q.E.D. ∎

:::

---

## V. Verification Checklist

- [x] All epsilon-delta arguments complete
- [x] All measure theory justified (Fubini, DCT conditions verified)
- [x] All constants explicit (C_Fisher, ε_clone, λ_LSI formulas given)
- [x] Edge cases: k=1 ✓, N→∞ ✓, boundary ✓
- [x] No circular reasoning
- [x] Framework dependencies verified

---

## VIII. Publication Readiness Assessment

**Mathematical Rigor**: 9/10
**Completeness**: 9/10
**Clarity**: 8/10
**Framework Consistency**: 10/10

**Overall**: **MEETS ANNALS OF MATHEMATICS STANDARD**

✅ Ready for submission after minor notation check

---

✅ Complete proof written to:
docs/source/1_euclidean_gas/proofs/proof_20251024_1630_thm_kl_convergence_euclidean.md

Length: 1,247 lines
Rigor: 9/10 (publication standard)
```

---

## Common Usage Patterns

### Pattern 1: Complete Workflow from Scratch

```
# Step 1: Create proof strategy
Load proof-sketcher.
Sketch proof for: thm-main-result
Document: docs/source/path/to/document.md

# Step 2: Expand to complete proof
Load theorem-prover.
Expand most recent sketch for: thm-main-result

# Step 3: Quality control
Load math-reviewer.
Review: docs/source/path/to/proofs/proof_*_thm_main_result.md
```

### Pattern 2: Handle Missing Lemmas Automatically

```
Load theorem-prover.

Expand proof for: thm-complex-result
Document: docs/source/path/document.md

# When prompted about missing lemmas:
Choose option 1: Sketch and prove lemmas first

# Agent will:
# 1. Detect lemma-A, lemma-B missing
# 2. Sketch and prove lemma-A
# 3. Sketch and prove lemma-B
# 4. Expand main theorem using proven lemmas
```

### Pattern 3: Expand Specific Steps Only

```
Load theorem-prover.

Expand steps 4-5 from sketch:
sketcher/sketch_20251024_1530_proof_theorem.md

Focus on: Add complete Fisher information derivation
Keep other steps as sketched
```

### Pattern 4: Incremental Expansion for Complex Proofs

```
# Session 1: Expand foundation
Expand steps 1-3 from sketch: sketcher/sketch_*.md

# Session 2: Expand technical core
Expand steps 4-6 from sketch: sketcher/sketch_*.md

# Session 3: Expand conclusion
Expand steps 7-9 from sketch: sketcher/sketch_*.md

# Then combine if needed
```

---

## Customization Options

### Rigor Depth

**Standard** (default):
```
Depth: standard
```
Normal Annals of Mathematics rigor (~2-3 hours)

**Maximum** (every epsilon):
```
Depth: maximum
```
Absolute maximum detail (~4-6 hours)

### Focus on Specific Technical Elements

```
Expand proof sketch: sketcher/sketch_*.md
Focus on:
- Step 4: Add complete epsilon-delta for all limits
- Step 5: Verify all Fubini conditions explicitly
- All steps: Track all constants with explicit formulas
```

---

## Real Example Walkthrough

I'll demonstrate with an actual proof:

```
Load theorem-prover agent from .claude/agents/theorem-prover.md

Expand proof sketch:
docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

**What the agent will do**:

1. **Read sketch** (lines 1-600 from sketch file)
2. **Extract**: 6 proof steps, chosen strategy (Lyapunov method), dependencies
3. **Check**: Identifies synergistic dissipation lemma (already proven in doc-03)
4. **Prepare prompts**: Creates 6 detailed expansion prompts (one per step)
5. **Submit to both AIs**: Gemini + GPT-5 expand all 6 steps in parallel
6. **Compare expansions**:
   - Step 1: Both complete (use Gemini's cleaner version)
   - Step 2: Both complete (use Gemini's structure)
   - Step 3: GPT-5 more detailed (use GPT-5's calculation)
   - Step 4: Gemini better Fubini, GPT-5 better ε-δ (synthesize)
   - Step 5: Both complete (use Gemini)
   - Step 6: Both complete (use Gemini)
7. **Synthesize**: Combine best elements
8. **Verify**: Check all framework deps, constants, edge cases
9. **Assess**: Rigor 9/10, meets publication standard
10. **Write**: `proofs/proof_20251024_1630_thm_kl_convergence_euclidean.md`

**Time**: ~2.5 hours

---

## Tips

1. **Always sketch first** - Theorem Prover needs a strategy to expand

2. **Check for missing lemmas** - Prove them before main theorem:
   ```
   # First
   Sketch and prove: lemma-X
   # Then
   Expand main theorem
   ```

3. **Use Math Reviewer after** - Validate expanded proof:
   ```
   # After Theorem Prover
   Load math-reviewer.
   Review: proofs/proof_*_{theorem}.md
   ```

4. **Iterate if needed**:
   ```
   # First expansion
   Expand proof (standard rigor) → assessment 7/10
   # Second expansion
   Re-expand steps 4, 7 (maximum rigor) → assessment 9/10
   ```

5. **Parallel expansion** for multiple theorems:
   ```
   Run 2 theorem-prover agents in parallel:
   Agent 1: Expand thm-A
   Agent 2: Expand thm-B
   ```

---

## Next Steps After Expansion

1. **Read Section II** (Expansion Comparison) - understand synthesis decisions
2. **Read Section IV** (Complete Proof) - verify rigor
3. **Read Section VIII** (Assessment) - check publication readiness
4. **If assessment ≥ 8/10**: Run Math Reviewer for final validation
5. **If assessment < 8/10**: Address remaining tasks listed
6. **Submit for publication** when Math Reviewer gives final approval

---

## Comparison: Workflow Stages

```
┌─────────────────┐
│ Proof Sketcher  │ → Strategy (3-7 steps, ~45 min)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Theorem Prover  │ → Complete proof (every detail, ~2-4 hours) ← YOU ARE HERE
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Math Reviewer   │ → Quality control (find gaps, ~45 min)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Publication     │ ✓
└─────────────────┘
```

---

That's it! Just copy-paste the simple usage above to expand your proof sketch.

For more details, see:
- Full agent definition: `.claude/agents/theorem-prover.md`
- Complete docs: `.claude/agents/theorem-prover-README.md`
- Framework context: `CLAUDE.md`
