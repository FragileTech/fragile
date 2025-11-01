# Proof Validation - Complete Workflow

## Prerequisites

- ✅ Theorems extracted from extract-and-refine workflow
- ✅ Document with theorem statements
- ✅ Access to Gemini 2.5 Pro and Codex (for dual-review)

---

## Complete Proof Development Pipeline

### Stage 1: Proof Sketching

#### Step 1.1: Identify Theorem

```bash
# List available theorems
ls docs/source/1_euclidean_gas/03_cloning/refined_data/theorems/

# Or query registry
python -c "
from fragile.proofs import load_registry_from_directory, MathematicalRegistry
registry = load_registry_from_directory(MathematicalRegistry, 'my_registry')
theorems = registry.get_all_theorems()
for thm in theorems[:10]:
    print(f'{thm.label}: {thm.name}')
"
```

#### Step 1.2: Run Proof Sketcher

**In Claude Code**:
```
Load proof-sketcher agent from .claude/agents/proof-sketcher.md

Sketch proof for: thm-keystone-principle
From document: docs/source/1_euclidean_gas/03_cloning.md
```

**Time**: ~5-15 minutes

**Agent output**:
```
✓ Analyzed theorem statement
✓ Identified dependencies: 3 objects, 2 axioms
✓ Generated proof strategy: 7 steps
✓ Wrote sketch to: reports/sketcher/sketch_20251028_1200_thm_keystone_principle.md
```

#### Step 1.3: Review Sketch

```bash
cat docs/source/1_euclidean_gas/03_cloning/reports/sketcher/sketch_20251028_1200_thm_keystone_principle.md
```

**Check for**:
- [ ] Strategy makes sense given theorem statement
- [ ] Required lemmas identified
- [ ] Steps in logical order
- [ ] No obvious gaps

**If issues**: Provide more context and re-run sketcher.

---

### Stage 2: Proof Expansion

#### Step 2.1: Run Theorem Prover

**In Claude Code**:
```
Load theorem-prover agent from .claude/agents/theorem-prover.md

Expand proof for: thm-keystone-principle
From sketch: reports/sketcher/sketch_20251028_1200_thm_keystone_principle.md
Document: docs/source/1_euclidean_gas/03_cloning.md
```

**Time**: ~20-45 minutes

**Agent output**:
```
✓ Loaded proof sketch: 7 SKETCHED steps
✓ Expanding step 1... [Gemini 2.5 Pro analysis]
✓ Expanding step 2...
  ...
✓ All steps EXPANDED
✓ Validated against framework axioms
✓ Wrote proof to: reports/proofs/proof_20251028_1230_thm_keystone_principle.md
```

#### Step 2.2: Review Expanded Proof

```bash
cat docs/source/1_euclidean_gas/03_cloning/reports/mathster/proof_20251028_1230_thm_keystone_principle.md
```

**Check for**:
- [ ] All steps fully expanded with derivations
- [ ] Mathematical notation correct
- [ ] References to framework axioms/lemmas
- [ ] Logical flow from hypothesis to conclusion

---

### Stage 3: Dual-Review Validation

#### Step 3.1: Run Math Reviewer

**In Claude Code**:
```
Load math-reviewer agent from .claude/agents/math-reviewer.md

Review: docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough
Focus on:
- Section 8: Keystone Principle proof
- Companion selection mechanism (§5.1)
```

**Time**: ~30-60 minutes

**What happens**:
1. Agent extracts relevant sections (~4-6 key sections)
2. Submits **identical prompts** to Gemini 2.5 Pro + Codex
3. Waits for both reviews (parallel execution)
4. Critically compares both reviews
5. Verifies claims against framework docs
6. Produces comprehensive report

**Agent output**:
```
✓ Extracted 5 sections (~1200 lines)
✓ Submitted to Gemini 2.5 Pro... [waiting]
✓ Submitted to Codex... [waiting]
✓ Received Gemini review: 8 issues found
✓ Received Codex review: 6 issues found
✓ Comparing reviews...
  - Consensus issues: 4
  - Contradictions: 1
  - Unique issues: 3
✓ Cross-validated against docs/glossary.md
✓ Wrote report to: reports/reviewer/review_20251028_1300_03_cloning.md
```

#### Step 3.2: Analyze Review Report

```bash
cat docs/source/1_euclidean_gas/03_cloning/reports/reviewer/review_20251028_1300_03_cloning.md
```

**Report sections**:
1. **Comparison Overview**: High-level statistics
2. **Issue Summary Table**: Compact view (sortable by severity)
3. **Detailed Analysis**: Each issue with:
   - Gemini's view
   - Codex's view
   - Claude's evidence-based judgment
   - Proposed fix with mathematical justification
4. **Implementation Checklist**: Action items
5. **Decision Points**: Where user input needed

**Priority order**:
1. CRITICAL issues (break the proof)
2. MAJOR issues (weaken claims)
3. MINOR issues (improve clarity)

---

### Stage 4: Targeted Verification

#### Step 4.1: Identify Claims to Verify

From review report, identify computational claims:
- Eigenvalue bounds
- Norm calculations
- Inequality derivations
- Constant values

#### Step 4.2: Run Math Verifier

**In Claude Code**:
```
Load math-verifier agent from .claude/agents/math-verifier.md

Verify: Eigenvalue gap bound ≥ 1 - exp(-C·t) in Lemma 5.3
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
Lines: 450-475
```

**Time**: ~15-30 minutes

**Agent output**:
```
✓ Extracted claim: λ_gap ≥ 1 - exp(-C·t)
✓ Generated SymPy validation script
✓ Executed verification... [symbolic computation]
✓ Result: VERIFIED ✅
✓ Created pytest test
✓ Wrote report to: reports/verifier/verification_20251028_1330_lemma_5_3.md
```

**Verification script** (`verification_lemma_5_3.py`):
```python
import sympy as sp

# Define symbols
t, C, lam_gap = sp.symbols('t C lambda_gap', positive=True, real=True)

# Claim: λ_gap ≥ 1 - exp(-C·t)
bound = 1 - sp.exp(-C * t)

# Verify bound properties
# Test: bound → 1 as t → ∞
limit = sp.limit(bound, t, sp.oo)
assert limit == 1, "Bound should approach 1"

# Test: bound increasing in t
derivative = sp.diff(bound, t)
assert sp.simplify(derivative) > 0, "Bound should increase with t"

print("✅ Verification passed")
```

#### Step 4.3: Run Verification Tests

```bash
# Run generated test
python docs/source/.../reports/verifier/verification_lemma_5_3.py

# Expected output:
# ✅ Verification passed

# Or run with pytest
pytest docs/source/.../reports/verifier/verification_lemma_5_3.py -v
```

---

## Implementation of Fixes

### Step 5.1: Review Proposed Fixes

From review report, for each CRITICAL/MAJOR issue:

```markdown
### Issue #1: Non-Circular Density Bound

**Proposed Fix**:
'''
Reformulate dynamics in terms of ψ(v) using Itô's lemma:
d(ψ(v)) = [ψ'(v) · (-γv - ∇U)] dt + [ψ'(v) · σ] dW_t
'''
```

### Step 5.2: Implement Fixes

Edit source markdown document:
```bash
vim docs/source/1_euclidean_gas/03_cloning.md
```

Make changes according to proposed fixes.

### Step 5.3: Re-run Review

After fixes, verify:
```
Load math-reviewer agent.

Review: docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough
Focus: Previously identified issues (§2.3.5, §6.3)
```

**Check**: Issues should be resolved or downgraded in severity.

---

## Iteration Loop

```
Sketch → Expand → Review
                    ↓
                Issues?
                    ↓
        Yes ← Fix → Re-review
         ↓
        No
         ↓
    Verify computations
         ↓
      Complete!
```

---

## Integration Points

### From Extract-and-Refine

After extracting theorems:
```bash
# Theorems available in refined_data/
ls docs/source/.../refined_data/theorems/

# Use proof-sketcher on specific theorem
```

### To Registry

After proof development:
```python
# Attach proof to theorem in registry
from fragile.proofs import attach_proof_to_theorem

theorem = registry.get_theorem_by_label('thm-keystone-principle')
proof = load_proof_from_file('reports/mathster/proof_*.md')
theorem = attach_proof_to_theorem(theorem, proof, validate=True)
```

---

## Time Summary

| Stage | Time | Can Skip? |
|-------|------|-----------|
| Sketching | ~5-15 min | No - foundation |
| Expansion | ~20-45 min | No - detail needed |
| Dual-review | ~30-60 min | No - quality control |
| Verification | ~15-30 min | Yes - only for computational claims |
| **Total** | **~1-2.5 hours** | Per theorem |

---

**Next**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues
