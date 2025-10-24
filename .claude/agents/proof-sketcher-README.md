# Proof Sketcher Agent - Documentation

**Version**: 1.0
**Created**: 2025-10-24
**Purpose**: Generate rigorous proof sketches for mathematical theorems using dual AI strategists

---

## Overview

The **Proof Sketcher** is an autonomous agent that generates detailed proof sketches for theorems in the Fragile mathematical framework by comparing proof strategies from Gemini 2.5 Pro and GPT-5 Pro.

### Key Distinction

| Feature | Math Reviewer | Proof Sketcher |
|---------|--------------|----------------|
| **Input** | Existing proofs | Theorem statements |
| **Goal** | Find errors | Create proof strategies |
| **Output** | Issue report with fixes | Detailed proof sketch |
| **Verification** | Against framework | Framework consistency + logical validity |
| **Use Case** | Quality control | Proof development |
| **When to Use** | After proof is written | Before proof is written |

---

## Core Capabilities

### Proof Strategy Generation
- Submits identical prompts to Gemini 2.5 Pro (strategic reasoning) + GPT-5 Pro (constructive proofs)
- Compares different proof approaches (direct, constructive, contradiction, induction, coupling, Lyapunov, compactness)
- Synthesizes optimal strategy based on framework constraints

### Framework Verification
- Cross-validates all dependencies against `docs/glossary.md` (741 entries)
- Ensures no circular reasoning (proof doesn't assume conclusion)
- Verifies all constants are defined and bounded
- Checks all preconditions of cited theorems are met

### Technical Analysis
- Identifies 1-3 most challenging technical points
- Proposes solutions with alternatives if main approach fails
- Tracks edge cases (k=1, N→∞, boundary conditions)
- Documents when additional lemmas are needed

### Structured Documentation
- Complete proof outline (3-7 major steps)
- Step-by-step sketch with justifications
- Framework dependency table
- Alternative approaches (not chosen)
- Expansion roadmap with time estimates

---

## How to Use

### Method 1: Single Theorem by Label

```
Load proof-sketcher agent.

Sketch proof for theorem: thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

### Method 2: Document with Focus

```
Load proof-sketcher.

Sketch proofs for: docs/source/1_euclidean_gas/06_convergence.md
Focus on: Foster-Lyapunov main theorem and drift lemmas
Depth: thorough
```

### Method 3: Multiple Theorems Explicitly

```
Load proof-sketcher.

Sketch proofs for: docs/source/1_euclidean_gas/04_wasserstein_contraction.md
Theorems: thm-wasserstein-contraction, lemma-coupling-construction
```

### Method 4: Complete Document

```
Load proof-sketcher.

Sketch all proofs for: docs/source/1_euclidean_gas/08_propagation_chaos.md
Depth: exhaustive
```

---

## Input Parameters

### Required
- **file_path**: Path to document containing theorems
  - Example: `docs/source/1_euclidean_gas/09_kl_convergence.md`

### Optional
- **theorems**: Specific theorem labels (comma-separated or list)
  - Example: `thm-main-result, lemma-key-bound, lemma-technical`
  - If omitted, agent will identify primary theorems automatically

- **focus_areas**: Topic descriptions instead of exact labels
  - Example: "LSI convergence proof, N-uniform bounds"
  - Useful when you don't know exact theorem labels

- **depth**: Level of coverage
  - `quick`: Main theorem only (~15 min)
  - `thorough`: Main + key lemmas (~45 min, **default**)
  - `exhaustive`: All theorems (~2 hours)

---

## Output Format

### File Location
```
docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

**Pattern**: `sketcher/sketch_{YYYYMMDD_HHMM}_proof_{document_name}.md`

### Sketch Structure

1. **Theorem Statement** (exact copy from source)
2. **Proof Strategy Comparison**
   - Gemini's approach with strengths/weaknesses
   - GPT-5's approach with strengths/weaknesses
   - Claude's synthesis (recommended strategy)
3. **Framework Dependencies** (verified against glossary)
   - Table of axioms used
   - Table of theorems used
   - Table of definitions used
   - Table of constants tracked
4. **Detailed Proof Sketch**
   - Overview (2-3 paragraphs)
   - Top-level outline (3-7 stages)
   - Step-by-step details with substeps
   - Justification for each step
5. **Technical Deep Dives** (1-3 challenges)
   - Why difficult
   - Proposed solution
   - Alternative if fails
6. **Proof Validation Checklist**
   - Logical completeness ✓
   - Hypothesis usage ✓
   - Framework consistency ✓
   - No circular reasoning ✓
7. **Alternative Approaches** (not chosen)
   - Description
   - Pros/cons
   - When to consider
8. **Open Questions & Future Work**
   - Remaining gaps
   - Conjectures
   - Extensions
9. **Expansion Roadmap**
   - Phase 1: Prove missing lemmas (with time estimate)
   - Phase 2: Fill technical details (with time estimate)
   - Phase 3: Add rigor (epsilon-delta, measure theory)
   - Phase 4: Review and validation
10. **Cross-References** (using `{prf:ref}` syntax)

---

## Example Output Preview

```markdown
# Proof Sketch for thm-kl-convergence-euclidean

## II. Proof Strategy Comparison

### Strategy A: Gemini's Approach
**Method**: Lyapunov method via entropy production
**Key Steps**:
1. Define Lyapunov function H(ρ) = D_KL(ρ || ρ_∞)
2. Compute entropy production dH/dt
3. Decompose into dissipation (kinetic) + expansion (cloning)
4. Show dissipation dominates expansion
5. Conclude exponential decay

**Strengths**:
- Direct application of framework's LSI machinery
- Tracks all constants explicitly

**Weaknesses**:
- Requires proving synergistic dissipation lemma first
- Complex Fisher information calculation

### Strategy B: GPT-5's Approach
**Method**: Coupling construction + relative entropy comparison
**Key Steps**:
1. Construct coupling of N-particle and QSD processes
2. Use coupling to bound W_2 distance
3. Apply Talagrand inequality: D_KL ≤ C·W_2²
4. Show coupling contracts in W_2
5. Conclude KL-convergence via inequality chain

**Strengths**:
- Intuitive probabilistic interpretation
- Reuses Wasserstein contraction from doc-04

**Weaknesses**:
- Talagrand constant C may depend on k
- Indirect path (W_2 → KL) loses sharpness

### Strategy Synthesis: Claude's Recommendation
**Chosen Method**: Lyapunov method (Gemini's approach)

**Rationale**:
- ✅ Direct proof of LSI (no inequality chain indirection)
- ✅ Explicit constant tracking ensures N-uniformity
- ✅ Framework already provides synergistic dissipation lemma (doc-03, Lemma 8.2)
- ⚠ Trade-off: More technical Fisher information calculation, but framework tools available

**Integration**:
- Steps 1-2: Gemini's entropy production setup
- Step 3: Use framework's decomposition (doc-03, Theorem 8.1)
- Step 4: GPT-5's insight about kinetic dominance condition
- Step 5: Gemini's exponential decay conclusion

## IV. Detailed Proof Sketch

### Step 1: Define Entropy Lyapunov Function

**Goal**: Establish H(ρ_t) = D_KL(ρ_t || ρ_∞) as Lyapunov function

**Substep 1.1**: Verify ρ_∞ is QSD
- **Justification**: Framework Theorem 4.3 (doc-02, thm-qsd-existence)
- **Why valid**: Axiom of Guaranteed Revival ensures uniqueness
- **Expected result**: ρ_∞ is unique invariant measure of P_Δt

**Substep 1.2**: Show H(ρ_t) ≥ 0 with equality iff ρ_t = ρ_∞
- **Justification**: Standard property of KL-divergence
- **Why valid**: ρ_t and ρ_∞ have common support (full alive set)
- **Expected result**: H is valid Lyapunov (non-negative, zero at equilibrium)

**Dependencies**:
- Uses: {prf:ref}`thm-qsd-existence` (doc-02)
- Requires: Axiom of Guaranteed Revival (axiom-revival, doc-01)

---

[... full 10-section sketch continues ...]
```

---

## Workflow

### What Happens When You Invoke

1. **Phase 1: Strategic Analysis** (~5-10 min)
   - Agent reads document
   - Uses Grep to find all `{prf:theorem}`, `{prf:lemma}` directives
   - Extracts theorem statements with Read
   - Identifies dependencies via Grep for `{prf:ref}`
   - Consults glossary.md for available framework results

2. **Phase 2: Prompt Preparation** (~5 min)
   - Constructs comprehensive prompt with:
     - Exact theorem statement
     - Available framework dependencies
     - Verification questions (specific to theorem type)
   - Same prompt sent to both strategists

3. **Phase 3: Dual Strategy Generation** (~10-15 min)
   - Submits to Gemini 2.5 Pro + GPT-5 Pro in parallel
   - Waits for both to return proof strategies
   - Parses outputs: approach, steps, lemmas, dependencies, challenges

4. **Phase 4: Critical Comparison** (~20-25 min)
   - Classifies agreements (consensus/complementary/contradictory)
   - Verifies every framework dependency in glossary.md
   - Checks for circular reasoning
   - Assesses technical feasibility
   - Synthesizes optimal strategy with justification

5. **Phase 5: Sketch Generation** (~10-15 min)
   - Formats complete proof sketch (10 sections)
   - Writes to `sketcher/sketch_{timestamp}_proof_{filename}.md`
   - Reports file location to user

**Total Time**:
- Quick: ~15 min (main theorem only)
- Thorough: ~45 min (main + lemmas, default)
- Exhaustive: ~2 hours (all theorems)

---

## Best Practices

### When to Use Proof Sketcher

✅ **Use For**:
- New theorems that need proofs
- Existing theorems with incomplete/unclear proofs
- Theorems where proof strategy is uncertain
- Complex theorems requiring careful planning
- Theorems with multiple possible approaches

❌ **Don't Use For**:
- Theorems with complete rigorous proofs (use Math Reviewer instead)
- Trivial lemmas with obvious proofs (just write the proof directly)
- Theorems outside the Fragile framework (agent won't have context)

### Tips for Best Results

1. **Be Specific About Focus**:
   ```
   Good: "Sketch proof for thm-kl-convergence-euclidean focusing on N-uniform constants"
   Better: "Sketch proof for thm-kl-convergence-euclidean, verify all constants are N-uniform"
   ```

2. **Choose Appropriate Depth**:
   - **Quick**: Initial exploration, sanity check if theorem is provable
   - **Thorough**: Standard workflow for most theorems
   - **Exhaustive**: Pre-publication verification of all proof strategies

3. **Provide Context** (if theorem is non-standard):
   ```
   Sketch proof for: docs/source/1_euclidean_gas/custom_theorem.md
   Context: This extends doc-09's LSI to time-inhomogeneous case
   ```

4. **Iterate on Sketches**:
   - First pass: Get overall strategy
   - Address gaps (prove missing lemmas)
   - Second pass: Re-sketch with new lemmas available
   - Expand to full proof

---

## Agent Guarantees

The Proof Sketcher ensures:

1. ✅ **Framework Consistency**: All dependencies verified in glossary.md
2. ✅ **No Circular Reasoning**: Proof steps don't assume conclusion
3. ✅ **Constant Tracking**: All mathematical constants defined and bounded
4. ✅ **Logical Completeness**: All parts of theorem statement addressed
5. ✅ **Actionable Steps**: Every step can be expanded to full rigorous proof
6. ✅ **Alternative Documentation**: Other approaches preserved for future use
7. ✅ **Independent Strategies**: Gemini + GPT-5 provide distinct perspectives

---

## Troubleshooting

### Problem: Agent Can't Find Theorem

**Symptoms**: "SKETCH FAILED - THEOREM NOT FOUND"

**Solutions**:
1. Check theorem label is exact (case-sensitive)
   ```bash
   grep -n "label:" docs/source/1_euclidean_gas/09_kl_convergence.md
   ```
2. Provide line number if label is non-standard:
   ```
   Sketch proof at line 450 of docs/source/...
   ```
3. Use focus areas instead of exact label:
   ```
   Focus on: main N-particle LSI convergence theorem
   ```

### Problem: Both Strategies Have Flaws

**Symptoms**: Sketch says "BOTH STRATEGIES HAVE CRITICAL ISSUES"

**What It Means**: Framework may not provide sufficient tools to prove theorem as stated

**Actions**:
1. Review agent's analysis of why both fail
2. Check if theorem statement needs weakening
3. Consider if framework needs extension (new axiom/lemma)
4. Consult user for domain expertise

### Problem: Missing Framework Dependencies

**Symptoms**: Sketch lists dependencies as "⚠ UNCERTAIN"

**What It Means**: Agent couldn't verify dependency in glossary.md

**Actions**:
1. Manually check if the dependency exists (may be under different label)
2. If truly missing, either:
   - Prove the missing result separately
   - Add to framework if it's a foundational fact
   - Find alternative proof approach that doesn't need it

### Problem: Sketch Takes Too Long

**Symptoms**: Agent running >1 hour for single theorem

**Causes**:
- Theorem has many dependencies (complex verification)
- Document is very large (slow extraction)
- Both strategists proposing complex approaches

**Solutions**:
- Use `depth: quick` for faster initial sketch
- Focus on specific theorem by label (not whole document)
- Try again later if MCP services are slow

---

## Advanced Usage

### Parallel Sketching of Multiple Documents

```
Run 3 proof-sketcher agents in parallel:

Agent 1: Sketch docs/source/1_euclidean_gas/04_wasserstein_contraction.md
         Focus: Main contraction theorem

Agent 2: Sketch docs/source/1_euclidean_gas/06_convergence.md
         Focus: Foster-Lyapunov

Agent 3: Sketch docs/source/1_euclidean_gas/09_kl_convergence.md
         Focus: N-particle LSI
```

All 3 will run independently and complete around the same time (~45 min each).

### Dependency-Driven Workflow

```
# Step 1: Sketch main theorem
Load proof-sketcher.
Sketch: docs/.../09_kl_convergence.md, theorem: thm-kl-convergence-euclidean

# Agent identifies: "Requires lemma-synergistic-dissipation (not yet proven)"

# Step 2: Sketch missing lemma
Sketch: docs/.../09_kl_convergence.md, theorem: lemma-synergistic-dissipation

# Agent provides proof strategy for lemma

# Step 3: Expand lemma proof (manually or with agent)
# Step 4: Re-sketch main theorem (now lemma is available)
```

### Comparison of Different Approaches

To explore multiple proof strategies:

```
Load proof-sketcher agent.

Sketch proof for: thm-wasserstein-contraction
Document: docs/.../04_wasserstein_contraction.md

Note: Interested in comparing coupling vs semigroup approach
```

Agent will document both in "Alternative Approaches" section even if one is chosen.

---

## Model Configuration

### Pinned Models (DO NOT CHANGE unless explicitly instructed):

- **Gemini**: `gemini-2.5-pro`
  - Strength: Strategic mathematical reasoning, theorem-level planning
  - Use case: High-level proof architecture

- **GPT-5**: `gpt-5` with `model_reasoning_effort=high`
  - Strength: Constructive proofs, explicit calculations with deep reasoning
  - Use case: Detailed step-by-step construction

### Why Two Models?

Different AI models have different strengths:
- **Gemini**: Better at abstract mathematical structure, identifying key insights
- **GPT-5**: Better at concrete construction, bound calculations
- **Synthesis**: Combining both provides more robust proof strategies

---

## Output File Management

### File Naming Convention
```
sketch_{YYYYMMDD_HHMM}_proof_{document_name}.md
```

**Examples**:
- `sketch_20251024_1530_proof_09_kl_convergence.md`
- `sketch_20251024_1545_proof_04_wasserstein_contraction.md`

### Directory Structure
```
docs/source/1_euclidean_gas/
├── 09_kl_convergence.md           # Original document
└── sketcher/                       # Proof sketches subdirectory
    ├── sketch_20251024_1530_proof_09_kl_convergence.md
    └── sketch_20251024_1600_proof_09_kl_convergence_v2.md  # Re-run after fixes
```

### Version Management

Multiple sketches for same theorem (iterative development):
- Timestamp prevents overwriting
- Compare sketches to see refinement
- Keep old sketches as alternatives

---

## Integration with Workflow

### Recommended Proof Development Workflow

```
1. Initial Draft → Proof Sketcher (thorough)
   ↓
2. Review sketch, identify missing lemmas
   ↓
3. Prove missing lemmas (Sketcher can help)
   ↓
4. Expand sketch to full proof (manual or assisted)
   ↓
5. Quality control → Math Reviewer (exhaustive)
   ↓
6. Fix issues identified by reviewer
   ↓
7. Final validation → Math Reviewer (thorough)
   ↓
8. Publication ready ✓
```

### Collaboration with Other Tools

- **Before Proof Sketcher**: Use Grep/Read to understand document structure
- **After Proof Sketcher**: Use formatting tools from `src/tools/` for LaTeX cleanup
- **Parallel with Proof Sketcher**: Can run Math Reviewer on different documents

---

## Limitations

### What Proof Sketcher CANNOT Do

❌ **Prove theorems automatically**: Only provides strategies, not complete proofs
❌ **Guarantee correctness**: Sketches require human expansion and verification
❌ **Handle theorems outside framework**: Requires Fragile framework context
❌ **Replace human insight**: Complex proofs need mathematician expertise
❌ **Verify computational results**: Numerical calculations need separate validation

### What Proof Sketcher CAN Do

✅ **Identify viable proof approaches**: Evaluates feasibility
✅ **Catch framework violations early**: Prevents invalid assumption chains
✅ **Document alternative strategies**: Preserves unexplored paths
✅ **Estimate proof complexity**: Roadmap provides time/difficulty estimates
✅ **Structure proof development**: Clear step-by-step plan

---

## FAQs

**Q: How is this different from Math Reviewer?**

A: Math Reviewer finds errors in *existing* proofs. Proof Sketcher creates *new* proof strategies before writing. Use Sketcher first (planning), Reviewer later (quality control).

**Q: Can I use custom models instead of Gemini/GPT-5?**

A: Models are pinned for consistency. You can override by explicitly requesting:
```
Use claude-opus-4 instead of gemini for proof strategy
```
But this is not recommended (agent is optimized for Gemini/GPT-5 pairing).

**Q: What if the sketch says "theorem may be unprovable"?**

A: This means the agent couldn't find a valid proof strategy with available framework tools. Options:
1. Weaken theorem statement
2. Add missing framework results (axioms/lemmas)
3. Get human expert opinion

**Q: Can I sketch proofs for theorems in other documents (not Fragile)?**

A: No. The agent is specialized for Fragile framework and uses `docs/glossary.md` for verification. For general theorems, use a general-purpose agent.

**Q: How do I know if a sketch is "good enough" to expand?**

A: Check the "Proof Validation Checklist" section (Section VI). If all items are ✓ and "Ready for Expansion: Yes", it's solid. If there are ⚠ warnings, address those first.

---

## Support

For issues or questions:
1. Check this README
2. See QUICKSTART guide for copy-paste examples
3. Consult `proof-sketcher.md` for agent internals
4. Check CLAUDE.md § Mathematical Proofing
5. Open issue: https://github.com/anthropics/claude-code/issues

---

**Version**: 1.0
**Last Updated**: 2025-10-24
**Maintainer**: Fragile Framework Team
