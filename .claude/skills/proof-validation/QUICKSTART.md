# Proof Validation - Quick Start

## TL;DR

Develop and validate proofs in 3 steps using dual-AI review.

```
# 1. Sketch proof strategy (~10 min)
Load proof-sketcher agent.
Sketch: thm-keystone-principle from docs/source/1_euclidean_gas/03_cloning.md

# 2. Expand full proof (~30 min)
Load theorem-prover agent.
Expand proof: thm-keystone-principle from sketch output

# 3. Dual-review (~45 min)
Load math-reviewer agent.
Review: docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough
Focus: Keystone Principle proof (Section 8)
```

**Output**: Complete reviewed proof in `reports/`

---

## Common Use Cases

### Use Case 1: Quick Proof Sketch

```
Load proof-sketcher agent.

Sketch proof for: thm-convergence-rate
From: docs/source/1_euclidean_gas/06_convergence.md
```

**Time**: ~5-10 minutes
**Output**: High-level proof strategy

---

### Use Case 2: Complete Proof Development

```
# Step 1: Sketch
Load proof-sketcher agent.
Sketch: thm-keystone-principle from docs/source/1_euclidean_gas/03_cloning.md

# Step 2: Expand
Load theorem-prover agent.
Expand: thm-keystone-principle from reports/sketcher/sketch_*.md

# Step 3: Review
Load math-reviewer agent.
Review: docs/source/1_euclidean_gas/03_cloning.md
Focus: Section 8 (Keystone Principle)
```

**Time**: ~1-1.5 hours total
**Output**: Fully developed and reviewed proof

---

### Use Case 3: Dual-Review Existing Proof

```
Load math-reviewer agent.

Review: docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md
Depth: thorough
Focus on:
- §2.3.5: Non-circular density bound
- §6.3-6.4: Telescoping identity for k-uniformity
```

**Time**: ~30-60 minutes
**Output**: Comprehensive review report with issue table

---

### Use Case 4: Verify Specific Calculation

```
Load math-verifier agent.

Verify: Eigenvalue gap bound in Lemma 5.3, lines 450-475
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

**Time**: ~15-30 minutes
**Output**: Executable verification script + validation report

---

## Verification

### After Sketching

```bash
# Check sketch created
ls -lh docs/source/.../reports/sketcher/

# Expected: sketch_YYYYMMDD_HHMM_thm_*.md file

# Review sketch
cat docs/source/.../reports/sketcher/sketch_*.md | head -50
```

### After Expansion

```bash
# Check proof created
ls -lh docs/source/.../reports/mathster/

# Expected: proof_YYYYMMDD_HHMM_thm_*.md file

# Check proof length (should be detailed)
wc -l docs/source/.../reports/mathster/proof_*.md
```

### After Review

```bash
# Check review report
ls -lh docs/source/.../reports/reviewer/

# Read issue summary
cat docs/source/.../reports/reviewer/review_*.md | grep -A 20 "## Issue Summary Table"
```

---

## Quick Troubleshooting

### Problem: Proof sketch too vague

**Solution**: Provide more context
```
Load proof-sketcher agent.

Sketch: thm-keystone-principle
Document: docs/source/1_euclidean_gas/03_cloning.md
Context: Uses Langevin dynamics + companion selection
Required lemmas: lem-markov-kernel-properties, lem-spectral-gap
```

### Problem: Review finds no issues

**Solution**: Increase depth or specify focus
```
Load math-reviewer agent.

Review: docs/source/.../document.md
Depth: exhaustive            # ← More thorough
Focus on: Specific proof sections that concern you
```

### Problem: Dual-review contradictions

**Solution**: This is good! Investigate both views
```
# When Gemini and Codex contradict:
# 1. Read both analyses carefully
# 2. Check framework documents (docs/glossary.md)
# 3. Verify claims manually
# 4. Claude will provide evidence-based judgment
```

---

## Parallel Processing

Review multiple documents simultaneously:

```
Launch 3 math-reviewer agents in parallel:

1. Review: docs/source/1_euclidean_gas/03_cloning.md (depth: thorough)
2. Review: docs/source/1_euclidean_gas/06_convergence.md (depth: quick)
3. Review: docs/source/2_geometric_gas/19_..._simplified.md (depth: exhaustive)

Provide 3 separate comprehensive reports.
```

All 3 will run independently and complete around the same time.

---

## Review Depth Guide

| Depth | Time | Sections Analyzed | Use When |
|-------|------|-------------------|----------|
| **Quick** | ~10 min | Abstract + main theorems | Sanity check |
| **Thorough** | ~30-45 min | Key sections + proofs | Standard (DEFAULT) |
| **Exhaustive** | ~1-2 hours | Complete document | Publication-ready |

---

## Next Steps After Review

1. **Read issue table** - Sorted by severity (CRITICAL → MAJOR → MINOR)
2. **Start with CRITICAL** - These break the proof
3. **Verify agent's verification** - Double-check claims against framework
4. **Implement fixes** - Use proposed fixes as starting point
5. **Re-run review** - Verify fixes with another pass

---

**Full Documentation**: [SKILL.md](./SKILL.md)
**Step-by-Step**: [WORKFLOW.md](./WORKFLOW.md)
**Issues**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
