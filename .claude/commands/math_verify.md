# Math Verify Command

Invoke the Math Verifier agent for autonomous symbolic validation of algebraic manipulations.

## Instructions

You are now acting as the Math Verifier agent. Follow the complete protocol defined in `.claude/agents/math-verifier.md`.

**CRITICAL**: You MUST read the agent definition file first:

```
Read: .claude/agents/math-verifier.md
```

Then execute the Math Verifier protocol on the document specified by the user.

## Expected Input Format

The user will provide input in one of these formats:

### Format 1: Validate Document
```
/math_verify docs/source/1_euclidean_gas/03_cloning.md
```

### Format 2: Validate Specific Theorem
```
/math_verify docs/source/1_euclidean_gas/03_cloning.md
Theorem: thm-keystone-lemma
```

### Format 3: Validate Proof Sketch
```
/math_verify sketcher/sketch_20251024_1530_proof_thm_keystone.md
```

### Format 4: Validate Complete Proof
```
/math_verify proofs/proof_20251024_1630_thm_keystone_lemma.md
```

### Format 5: With Depth and Focus
```
/math_verify docs/source/1_euclidean_gas/03_cloning.md
Depth: exhaustive
Focus: Variance decompositions, logarithmic bounds
```

## Parameters

- **file_path** (required): Path to document/sketch/proof to validate
- **theorem_label** (optional): Specific theorem to focus on
- **focus_areas** (optional): Specific algebraic categories (variance, logarithms, quadratic forms, etc.)
- **depth** (optional): `quick` | `thorough` (default) | `exhaustive`

## Agent Protocol

After reading the agent definition, you MUST follow this autonomous workflow:

### Phase 1: Document Analysis (10-15%)
1. Validate file exists and map structure
2. Extract all theorems/lemmas with Grep
3. Identify algebraic claims (equations, derivations, identities)
4. Categorize by validation type (variance, log, quadratic, etc.)

### Phase 2: Framework Integration (10%)
1. Extract symbols from docs/glossary.md
2. Extract constants and bounds from document
3. Build symbol mapping for validation code

### Phase 3: Dual Code Generation (30-40%)
1. For EACH algebraic claim, construct comprehensive prompt
2. Submit to BOTH in parallel:
   - Gemini 2.5 Pro (model="gemini-2.5-pro")
   - GPT-5 Pro (model="gpt-5-pro")
3. Wait for both responses
4. Parse and score both code versions

### Phase 4: Validation Execution (10-15%)
1. Synthesize best code from both AIs
2. Write validation scripts to `src/proofs/{doc_name}/{theorem_label}.py`
3. Execute validation scripts and collect results
4. Track passed/failed/error status

### Phase 5: Report Generation (15-20%)
1. Generate comprehensive report following template
2. Include: category breakdown, detailed results, code comparison
3. Write to file: `verifier/verification_{timestamp}_{filename}.md`
4. Display summary to user

## Output

The agent will:
1. Execute complete symbolic validation autonomously
2. Write validation scripts to `src/proofs/{doc_name}/`
3. Write comprehensive report to `verifier/`
4. Display summary with pass/fail statistics

**File Locations**:
- **Validation Scripts**: `src/proofs/{doc_name}/{theorem_label}.py`
- **Verification Report**: `{document_dir}/verifier/verification_{YYYYMMDD_HHMM}_{doc_name}.md`

## Quality Guarantees

- ✅ Dual AI code generation for robustness
- ✅ Framework symbol integration from glossary.md
- ✅ Pytest-compatible test suite
- ✅ Executable validation scripts (no placeholders)
- ✅ Clear separation: algebraic (✓) vs semantic (⚠️) steps
- ✅ Pass/fail status for each validation

## Notes

- Agent runs autonomously (no interruptions)
- Expected runtime: 20 min (quick) / 45 min (thorough) / 2 hours (exhaustive)
- Generates pytest-compatible scripts for continuous validation
- Multiple instances can run in parallel on different documents
- Complements Math Reviewer (algebraic validation before semantic review)

## Integration with Proof Pipeline

**Dual Validation Workflow** (RECOMMENDED):
```
Proof Sketcher → Math Verifier (strategy) → Theorem Prover → Math Verifier (proof) → Math Reviewer
```

**Quick Validation Workflow**:
```
Proof Sketcher → Theorem Prover → Math Verifier → Math Reviewer
```

**Standalone Validation** (existing documents):
```
Math Verifier → Math Reviewer
```

---

**Now begin the Math Verifier protocol for the document provided by the user.**
