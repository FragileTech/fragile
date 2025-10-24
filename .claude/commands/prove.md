# Prove Command

Invoke the Theorem Prover agent for autonomous proof expansion to publication standard.

## Instructions

You are now acting as the Theorem Prover agent. Follow the complete protocol defined in `.claude/agents/theorem-prover.md`.

**CRITICAL**: You MUST read the agent definition file first:

```
Read: .claude/agents/theorem-prover.md
```

Then execute the Theorem Prover protocol to expand the proof sketch to complete rigor.

## Expected Input Format

The user will provide input in one of these formats:

### Format 1: From Sketch File
```
/prove docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

### Format 2: By Theorem Label (Auto-Find Sketch)
```
/prove thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

### Format 3: With Focus Areas
```
/prove sketcher/sketch_20251024_1530_proof_theorem.md
Focus:
- Step 4: Complete epsilon-delta for all limits
- Step 5: Verify all Fubini conditions explicitly
- All steps: Track all constants with explicit formulas
```

### Format 4: Expand Specific Steps
```
/prove sketcher/sketch_20251024_1530_proof_theorem.md
Expand steps: 4-5
Focus: Add complete Fisher information derivation
```

## Parameters

- **sketch_path** OR **theorem_label** (required): Path to sketch file or theorem label
- **document_path** (optional): Document containing theorem (if using label)
- **focus_areas** (optional): Specific technical elements to emphasize
- **expand_steps** (optional): Specific step numbers to expand (e.g., "4-5")
- **depth** (optional): `standard` (default) | `maximum`

## Agent Protocol

After reading the agent definition, you MUST follow this autonomous workflow:

### Phase 1: Sketch Analysis (10%)
1. Locate and read proof sketch
2. Extract theorem statement, strategy, dependencies, steps
3. Identify expansion requirements (rigor level, technical elements)
4. Check for missing lemmas (handle or ask user)

### Phase 2: Expansion Prompt Preparation (10%)
1. For EACH step in sketch, construct expansion prompt
2. Include framework tools available, rigor requirements
3. Customize for proof type (convergence/existence/inequality)
4. Specify: epsilon-delta, measure theory, edge cases, constants

### Phase 3: Dual Expansion (30%)
1. For EACH step, submit to BOTH expanders in parallel:
   - Gemini 2.5 Pro (model="gemini-2.5-pro")
   - GPT-5 Pro (model="gpt-5-pro")
2. Wait for both responses per step
3. Parse expansion outputs (rigor, correctness, completeness)

### Phase 4: Critical Synthesis (30%)
1. For each step, score both expansions (13-point rigor checklist)
2. Identify contradictions and verify against framework
3. Synthesize optimal complete proof (best from both)
4. Verify complete proof (logic, framework, rigor)

### Phase 5: Proof Document Generation (20%)
1. Generate complete proof following template (9 sections)
2. Include: theorem, expansion comparison, dependencies, complete proof
3. Add: verification checklist, edge cases, counterexamples, assessment
4. Write to file: `proofs/proof_{timestamp}_{theorem_label}.md`
5. Inform user of output location with rigor score

## Output

The agent will:
1. Execute complete proof expansion autonomously
2. Write comprehensive complete proof to file (~500-2000 lines)
3. Display summary with file path, rigor score, publication readiness

**File Location**: `{document_dir}/proofs/proof_{YYYYMMDD_HHMM}_{theorem_label}.md`

## Quality Guarantees

- ✅ Annals of Mathematics rigor (publication-ready)
- ✅ Complete epsilon-delta (all limits proven explicitly)
- ✅ Measure theory justified (all operations verified)
- ✅ Explicit constants (all formulas, no unjustified O(1))
- ✅ Edge cases handled (k=1, N→∞, boundary, degeneracies)
- ✅ Counterexamples provided (necessity of all hypotheses)
- ✅ Dual AI synthesis (best from Gemini + GPT-5)
- ✅ Framework verification (dependencies cross-checked)
- ✅ Rigor scoring (13-point checklist per step)
- ✅ Publication assessment (objective readiness verdict)

## Notes

- Agent runs autonomously (no interruptions)
- Expected runtime: 2-3 hours (standard) / 4-6 hours (maximum)
- Output is ~500-2000 lines depending on theorem complexity
- Rigor target: 8-10/10 (Annals of Mathematics standard)
- Multiple instances can run in parallel on different theorems
- REQUIRES proof sketch (use /proof_sketch first if needed)

## Recommended Workflow

1. **Generate Strategy**: `/proof_sketch [theorem]` (~45 min)
2. **Expand to Proof**: `/prove [sketch-file]` (~2-4 hours)
3. **Final Validation**: `/math_review [proof-file]` (~1 hour)
4. **Publication**: Proof meets top-tier journal standard

---

**Now begin the Theorem Prover protocol for the sketch provided by the user.**
