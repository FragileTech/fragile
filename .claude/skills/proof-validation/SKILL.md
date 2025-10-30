---
name: proof-validation
description: Develop, expand, and validate mathematical proofs using dual-AI review (Gemini + Codex). Use when sketching proof strategies, expanding proofs, reviewing mathematical rigor, or verifying computational claims in proofs.
---

# Proof Validation Skill

## Purpose

Complete pipeline for developing, expanding, and validating mathematical proofs using dual-AI review (Gemini 2.5 Pro + Codex).

**Input**: Theorems from extract-and-refine workflow
**Output**: Fully developed, reviewed, and verified proofs
**Pipeline**: Proof Sketcher → Theorem Prover → Math Reviewer → Math Verifier

---

## Agents Involved

| Agent | Role | Input | Output |
|-------|------|-------|--------|
| **proof-sketcher** | Generate proof strategy | Theorem statement | Proof sketch (SKETCHED steps) |
| **theorem-prover** | Expand proof details | Proof sketch | Full proof (EXPANDED steps) |
| **math-reviewer** | Dual-review rigor | Document/proof | Review report with issues |
| **math-verifier** | Validate correctness | Specific claims | Verification report |

---

## Complete Workflow

### Stage 1: Proof Sketching

**Purpose**: Generate high-level proof strategy

```
Load proof-sketcher agent.

Sketch proof for: thm-keystone-principle
From document: docs/source/1_euclidean_gas/03_cloning.md
```

**What it does**:
- Analyzes theorem statement and dependencies
- Generates proof strategy (5-10 steps)
- Creates ProofBox with SKETCHED steps
- Identifies required lemmas and axioms

**Output**:
```
docs/source/1_euclidean_gas/03_cloning/reports/sketcher/
└── sketch_20251028_1200_thm_keystone_principle.md
```

**Time**: ~5-15 minutes (depends on theorem complexity)

---

### Stage 2: Proof Expansion

**Purpose**: Fill in mathematical details

```
Load theorem-prover agent.

Expand proof: thm-keystone-principle
From sketch: reports/sketcher/sketch_20251028_1200_thm_keystone_principle.md
```

**What it does**:
- Takes proof sketch as input
- Expands each SKETCHED step with full derivations
- Uses Gemini 2.5 Pro for mathematical reasoning
- Validates each step against framework axioms
- Creates complete proof with all details

**Output**:
```
docs/source/1_euclidean_gas/03_cloning/reports/proofs/
└── proof_20251028_1230_thm_keystone_principle.md
```

**Time**: ~20-45 minutes (depends on proof length)

---

### Stage 3: Dual-Review Validation

**Purpose**: Critical review for rigor and correctness

```
Load math-reviewer agent.

Review: docs/source/1_euclidean_gas/03_cloning.md
Depth: thorough
Focus on: Keystone Principle proof (Section 8)
```

**What it does**:
1. Extracts relevant sections strategically
2. Submits **identical prompts** to Gemini 2.5 Pro + Codex in parallel
3. Waits for both reviews to complete
4. **Critically compares** both reviews:
   - Consensus issues (high confidence)
   - Contradictions (investigates)
   - Cross-validates against framework docs
5. Makes **evidence-based judgments** about correctness
6. Produces comprehensive report

**Output**:
```
docs/source/1_euclidean_gas/03_cloning/reports/reviewer/
└── review_20251028_1300_03_cloning.md
```

**Report includes**:
- Issue summary table (compact overview)
- Detailed issue analysis with severity ratings
- Proposed fixes with mathematical justification
- Implementation checklist
- Your decision points

**Time**: ~30-60 minutes (depends on document size and depth)

---

### Stage 4: Targeted Verification

**Purpose**: Verify specific mathematical claims

```
Load math-verifier agent.

Verify: Lemma 4.2 bound derivation in Section 8.3
Document: docs/source/1_euclidean_gas/03_cloning.md
```

**What it does**:
- Focuses on specific mathematical claim
- Generates symbolic computation validation
- Creates executable Python/SymPy scripts
- Runs verification and reports results
- Provides pytest-compatible tests

**Output**:
```
docs/source/1_euclidean_gas/03_cloning/reports/verifier/
└── verification_20251028_1330_lemma_4_2.md
```

**Time**: ~15-30 minutes per claim

---

## Dual-Review Protocol

### Why Dual Review?

**Problem**: Single AI can hallucinate or miss subtle issues

**Solution**: Independent parallel reviews + critical comparison

| Feature | Single AI | Dual AI (Gemini + Codex) |
|---------|-----------|---------------------------|
| **Hallucination risk** | High | Low (cross-validated) |
| **Coverage** | Single perspective | Diverse viewpoints |
| **Confidence** | Uncertain | Consensus = high confidence |
| **Error detection** | May miss | Contradictions flagged |

### How It Works

1. **Identical prompts** sent to both AIs simultaneously
2. **Independent analysis** (no communication between AIs)
3. **Critical comparison**:
   - **Consensus** (both agree) → High confidence, prioritize
   - **Contradiction** (disagree) → Investigate, verify manually
   - **Unique** (only one identifies) → Medium confidence, verify

4. **Evidence-based judgment**: Always verify claims against framework docs

5. **Final report**: Synthesizes both views with Claude's analysis

---

## Best Practices

### 1. Start with Proof Sketch

Don't jump directly to full proofs:

```
✅ Correct: Sketch → Review sketch → Expand → Review full proof
❌ Wrong: Directly write full proof → Hard to revise if strategy wrong
```

### 2. Use Appropriate Review Depth

| Depth | Time | Use When |
|-------|------|----------|
| **Quick** | ~10 min | Sanity checks, minor changes |
| **Thorough** | ~30-45 min | Standard review (DEFAULT) |
| **Exhaustive** | ~1-2 hours | Critical proofs, publication |

### 3. Always Use Dual-Review for Important Proofs

```
Load math-reviewer agent.  # Automatically uses dual-review

Review: docs/source/.../document.md
Depth: thorough
```

**Never skip dual-review** for theorems going into publications.

### 4. Verify Computational Claims

For proofs with calculations:

```
Load math-verifier agent.

Verify: Eigenvalue bound computation in Lemma 5.3
```

Generates executable verification scripts.

### 5. Iterate Based on Feedback

After review:
1. Read issue table (sorted by severity)
2. Start with CRITICAL issues
3. Implement proposed fixes
4. Re-run review to verify fixes

---

## Integration with Other Workflows

### From Extract-and-Refine

After extracting theorems:

```bash
# List available theorems
ls docs/source/.../refined_data/theorems/

# Pick theorem for proof development
# Load proof-sketcher agent
```

### From Registry

Query registry for theorems to prove:

```python
from fragile.proofs import load_registry_from_directory, MathematicalRegistry

registry = load_registry_from_directory(MathematicalRegistry, 'my_registry')
theorems = registry.get_all_theorems()

# Find theorems without proofs
unproven = [thm for thm in theorems if not thm.has_proof()]
print(f'{len(unproven)} theorems need proofs')
```

---

## Output Format

### Proof Sketch

```markdown
# Proof Sketch: Keystone Principle

## Strategy

Use Langevin relaxation + companion selection to prove...

## Steps

1. [SKETCHED] Establish Markov kernel properties
   - Show φ_kin is irreducible
   - Verify detailed balance

2. [SKETCHED] Prove exponential convergence
   - Use spectral gap argument
   - Apply Poincaré inequality
   ...
```

### Review Report

```markdown
# Dual Review Summary for 03_cloning.md

## Comparison Overview
- Consensus Issues: 4 (both agree)
- Gemini-Only Issues: 1
- Codex-Only Issues: 2
- Contradictions: 1

## Issue Summary Table
| # | Issue | Severity | Location | Gemini | Codex | Claude | Status |
|---|-------|----------|----------|--------|-------|--------|--------|
| 1 | Non-circular density | CRITICAL | §2.3.5 | ✓ | ✓ | ✅ Verified | Action required |

## Detailed Analysis

### Issue #1: Non-Circular Density Bound (CRITICAL)
- **Gemini**: "Insufficient proof that doc-13 avoids density assumptions"
- **Codex**: "Velocity squashing doesn't make domain compact - fatal flaw"
- **My Assessment**: ✅ VERIFIED CRITICAL

**Evidence**: [verified against framework docs]

**Proposed Fix**:
```
Reformulate dynamics in terms of ψ(v) using Itô's lemma
```
```

---

## Performance Tips

| Task | Time | Notes |
|------|------|-------|
| Proof sketch | ~5-15 min | Fast, iterate freely |
| Proof expansion | ~20-45 min | Slow, get sketch right first |
| Review (thorough) | ~30-60 min | Standard depth |
| Verification | ~15-30 min | Per claim |

**Recommendation**: Invest time in good proof sketch before expansion.

---

## Related Documentation

- **Agent Definitions**:
  - `.claude/agents/proof-sketcher.md`
  - `.claude/agents/theorem-prover.md`
  - `.claude/agents/math-reviewer.md`
  - `.claude/agents/math-verifier.md`
- **Quick Start**: [QUICKSTART.md](./QUICKSTART.md)
- **Detailed Workflow**: [WORKFLOW.md](./WORKFLOW.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- **CLAUDE.md**: Mathematical rigor requirements
- **GEMINI.md**: Gemini review protocol

---

## Version History

- **v1.0.0** (2025-10-28): Initial proof-validation skill
  - Consolidated proof-sketcher, theorem-prover, math-reviewer, math-verifier
  - Documented dual-review protocol
  - Added integration with extract-and-refine and registry-management

---

**Next**: See [QUICKSTART.md](./QUICKSTART.md) for copy-paste commands.
