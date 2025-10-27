# Running Agents in Parallel

**All 5 agents are now registered** and available as `@agent-` calls for parallel execution!

---

## ✅ Registered Agents

| Agent | Color | Purpose | Runtime |
|-------|-------|---------|---------|
| `@agent-document-parser` | 🔵 Cyan | Extract mathematical content | ~30 sec |
| `@agent-proof-sketcher` | 🟢 Green | Generate proof strategies | ~45 min |
| `@agent-math-verifier` | 🟠 Orange | Validate algebra with sympy | ~30 min |
| `@agent-theorem-prover` | 🟣 Purple | Expand to complete proofs | ~2-4 hrs |
| `@agent-math-reviewer` | 🔵 Blue | Dual-review quality control | ~45 min |

---

## Basic Usage

You can now invoke agents in two ways:

### Method 1: Direct `@agent-` Mention
```
@agent-math-reviewer review this document:
docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md
```

### Method 2: Via Task Tool (More Control)
```python
Task(
    subagent_type="math-reviewer",
    description="Review geometric gas regularity",
    prompt="""
    Review: docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md
    Depth: thorough
    Focus: Non-circularity, k-uniformity
    """
)
```

---

## Parallel Execution Examples

### Example 1: Review 3 Documents Simultaneously

```
I need you to launch 3 math-reviewer agents in parallel to review these documents:

1. @agent-math-reviewer review docs/source/1_euclidean_gas/03_cloning.md (focus: Keystone Principle)
2. @agent-math-reviewer review docs/source/2_geometric_gas/11_geometric_gas.md (depth: thorough)
3. @agent-math-reviewer review docs/source/2_geometric_gas/13_geometric_gas_c3_regularity.md (depth: exhaustive)

Run all three simultaneously and provide separate reports.
```

### Example 2: Complete Proof Pipeline in Parallel

```
Launch the following agents in parallel on different theorems:

1. @agent-proof-sketcher sketch proof for thm-kl-convergence-euclidean
   Document: docs/source/1_euclidean_gas/09_kl_convergence.md

2. @agent-proof-sketcher sketch proof for thm-wasserstein-contraction
   Document: docs/source/1_euclidean_gas/04_wasserstein_contraction.md

3. @agent-proof-sketcher sketch proof for thm-foster-lyapunov
   Document: docs/source/1_euclidean_gas/06_convergence.md

Generate proof strategies for all three simultaneously.
```

### Example 3: Dual Validation (Algebra + Semantic)

```
Run these two validation agents in parallel on the same document:

1. @agent-math-verifier validate algebraic manipulations in:
   docs/source/1_euclidean_gas/03_cloning.md
   Focus: Variance decompositions, logarithmic bounds

2. @agent-math-reviewer review mathematical rigor in:
   docs/source/1_euclidean_gas/03_cloning.md
   Focus: Proof structure, framework consistency

Compare both validation results.
```

### Example 4: Multi-Stage Pipeline

```
Launch a 5-stage proof development pipeline in parallel:

Stage 1 (3 agents):
1. @agent-proof-sketcher for thm-A in doc1.md
2. @agent-proof-sketcher for thm-B in doc2.md
3. @agent-proof-sketcher for thm-C in doc3.md

Stage 2 (3 agents - after Stage 1 completes):
4. @agent-math-verifier validate sketch A
5. @agent-math-verifier validate sketch B
6. @agent-math-verifier validate sketch C

Stage 3 (3 agents - after Stage 2 completes):
7. @agent-theorem-prover expand sketch A
8. @agent-theorem-prover expand sketch B
9. @agent-theorem-prover expand sketch C

Run stages in sequence but agents within each stage in parallel.
```

### Example 5: Document Analysis Pipeline

```
Analyze document structure with multiple agents in parallel:

1. @agent-document-parser parse:
   docs/source/1_euclidean_gas/03_cloning.md
   Mode: both

2. @agent-math-reviewer review:
   docs/source/1_euclidean_gas/03_cloning.md
   Depth: quick (sanity check)

3. @agent-math-verifier validate:
   docs/source/1_euclidean_gas/03_cloning.md
   Depth: quick

Generate complete analysis from all three perspectives.
```

---

## Task Tool Syntax (Advanced)

For maximum control, use the Task tool directly:

```python
# Single agent with custom parameters
Task(
    subagent_type="theorem-prover",
    description="Expand KL convergence proof",
    prompt="""
    Expand proof sketch:
    docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md

    Depth: maximum
    Focus:
    - Step 4: Complete all epsilon-delta arguments
    - All steps: Track all constants explicitly
    """
)

# Multiple agents in parallel (single message with multiple Task calls)
Task(subagent_type="math-reviewer", ...)
Task(subagent_type="math-verifier", ...)
Task(subagent_type="proof-sketcher", ...)
```

---

## Best Practices for Parallel Execution

### 1. Independent Tasks
✅ **Good** - Each agent works on different documents/theorems:
```
Agent 1: Review doc1.md
Agent 2: Review doc2.md
Agent 3: Review doc3.md
```

❌ **Bad** - Agents depend on each other's output:
```
Agent 1: Sketch proof
Agent 2: Expand sketch from Agent 1  # Must wait!
```

### 2. Resource Management
- **CPU-intensive**: document-parser (~30 sec)
- **API-intensive**: math-reviewer, proof-sketcher, math-verifier, theorem-prover
- **Long-running**: theorem-prover (~2-4 hours)

**Recommended limits**:
- Max 3 theorem-prover agents in parallel (very long-running)
- Max 5 math-reviewer/verifier agents in parallel (API rate limits)
- Unlimited document-parser (fast, local)

### 3. Staging
For complex pipelines, run in stages:
```
Stage 1: Sketch all proofs (parallel)
         → Wait for completion
Stage 2: Verify all sketches (parallel)
         → Wait for completion
Stage 3: Expand all proofs (parallel)
```

---

## Output Management

Each agent writes to its own location:
```
docs/source/1_euclidean_gas/
├── data/
│   └── extraction_inventory.json       # document-parser
├── sketcher/
│   ├── sketch_20251024_1530_A.md       # proof-sketcher #1
│   ├── sketch_20251024_1531_B.md       # proof-sketcher #2
│   └── sketch_20251024_1532_C.md       # proof-sketcher #3
├── verifier/
│   ├── verification_20251024_1430_A.md # math-verifier #1
│   └── verification_20251024_1431_B.md # math-verifier #2
├── proofs/
│   ├── proof_20251024_1630_thm_A.md    # theorem-prover #1
│   └── proof_20251024_1631_thm_B.md    # theorem-prover #2
└── reviewer/
    ├── review_20251024_1430_doc1.md    # math-reviewer #1
    └── review_20251024_1431_doc2.md    # math-reviewer #2
```

**No file conflicts** - each agent creates timestamped files.

---

## Monitoring Parallel Agents

Since agents run autonomously, you can:

1. **Check progress** - Look for intermediate outputs in directories
2. **View logs** - Each agent reports its progress
3. **Aggregate results** - Collect all outputs after completion
4. **Compare reports** - Review findings across parallel instances

---

## Common Patterns

### Pattern 1: Batch Review
```
Review all documents in a chapter simultaneously:
- Agent 1: docs/source/1_euclidean_gas/02_euclidean_gas.md
- Agent 2: docs/source/1_euclidean_gas/03_cloning.md
- Agent 3: docs/source/1_euclidean_gas/04_convergence.md
- Agent 4: docs/source/1_euclidean_gas/05_mean_field.md
```

### Pattern 2: Multi-Theorem Development
```
Sketch proofs for all theorems in a document:
- Agent 1: thm-main-convergence
- Agent 2: lemma-drift-condition
- Agent 3: lemma-moment-bound
- Agent 4: thm-rate-explicit
```

### Pattern 3: Comprehensive Validation
```
Triple-validation on critical document:
- Agent 1: math-verifier (algebra check)
- Agent 2: math-reviewer (semantic check)
- Agent 3: document-parser (structure check)
```

---

## Quick Reference

| Want to... | Use... | Parallel? |
|------------|--------|-----------|
| Review multiple documents | `@agent-math-reviewer` × N | ✅ Yes |
| Sketch multiple proofs | `@agent-proof-sketcher` × N | ✅ Yes |
| Validate multiple proofs | `@agent-math-verifier` × N | ✅ Yes |
| Expand multiple sketches | `@agent-theorem-prover` × N | ✅ Yes (max 3) |
| Parse multiple documents | `@agent-document-parser` × N | ✅ Yes |
| Complete proof pipeline | Sequential stages, parallel within stages | ⚠️ Staged |

---

## Testing Agent Registration

Try this simple test:
```
@agent-math-reviewer review docs/source/1_euclidean_gas/03_cloning.md
Depth: quick
```

You should see the agent load and start the dual-review protocol.

---

**All 5 agents registered and ready for parallel execution!**

Type `@agent-` and see the autocomplete suggestions, or use `/agents` to list all available agents.
