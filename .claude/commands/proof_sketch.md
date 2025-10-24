# Proof Sketch Command

Invoke the Proof Sketcher agent for autonomous proof strategy generation.

## Instructions

You are now acting as the Proof Sketcher agent. Follow the complete protocol defined in `.claude/agents/proof-sketcher.md`.

**CRITICAL**: You MUST read the agent definition file first:

```
Read: .claude/agents/proof-sketcher.md
```

Then execute the Proof Sketcher protocol for the theorem(s) specified by the user.

## Expected Input Format

The user will provide input in one of these formats:

### Format 1: Single Theorem by Label
```
/proof_sketch thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

### Format 2: Document with Focus
```
/proof_sketch docs/source/1_euclidean_gas/06_convergence.md
Focus: Foster-Lyapunov main theorem, drift lemmas
```

### Format 3: Multiple Theorems
```
/proof_sketch docs/source/1_euclidean_gas/04_wasserstein_contraction.md
Theorems: thm-wasserstein-contraction, lemma-coupling-construction
```

### Format 4: Complete Document
```
/proof_sketch docs/source/1_euclidean_gas/08_propagation_chaos.md
Depth: exhaustive
```

## Parameters

- **file_path** OR **theorem_label** (required): Document path or specific theorem label
- **theorems** (optional): Comma-separated list of theorem labels
- **focus_areas** (optional): Topic descriptions
- **depth** (optional): `quick` | `thorough` (default) | `exhaustive`

## Agent Protocol

After reading the agent definition, you MUST follow this autonomous workflow:

### Phase 1: Document Analysis (10-15%)
1. Validate and locate document
2. Map theorem structure (grep for {prf:theorem}, {prf:lemma})
3. Extract theorem statements with context
4. Identify dependencies from framework
5. Prioritize based on depth setting

### Phase 2: Prompt Preparation (10%)
1. Construct comprehensive identical prompt for both strategists
2. Include theorem statement, framework context, verification tasks
3. Customize for theorem type (convergence/existence/inequality)

### Phase 3: Dual Strategy Generation (15%)
1. Submit to BOTH strategists in parallel:
   - Gemini 2.5 Pro (model="gemini-2.5-pro")
   - GPT-5 Pro (model="gpt-5-pro")
2. Wait for both responses
3. Parse both strategy outputs

### Phase 4: Strategy Synthesis (35%)
1. Classify agreements and disagreements
2. Verify framework dependencies against glossary.md
3. Assess technical validity of each approach
4. Synthesize optimal proof strategy with evidence-based judgment

### Phase 5: Sketch Document Generation (15%)
1. Generate complete proof sketch following template (10 sections)
2. Include: theorem, comparison, dependencies, detailed steps, challenges, alternatives
3. Write to file: `sketcher/sketch_{timestamp}_proof_{doc_name}.md`
4. Inform user of output location

## Output

The agent will:
1. Execute complete proof sketching autonomously
2. Write comprehensive sketch to file
3. Display summary with file path

**File Location**: `{document_dir}/sketcher/sketch_{YYYYMMDD_HHMM}_proof_{doc_name}.md`

## Quality Guarantees

- ✅ Framework consistency (all dependencies verified in glossary.md)
- ✅ No circular reasoning (proof steps don't assume conclusion)
- ✅ Constant tracking (all mathematical constants defined and bounded)
- ✅ Logical completeness (all parts of theorem addressed)
- ✅ Actionable steps (expandable to full rigorous proof)
- ✅ Alternative documentation (other approaches preserved)
- ✅ Independent strategies (Gemini + GPT-5 distinct perspectives)

## Notes

- Agent runs autonomously (no interruptions)
- Expected runtime: 15 min (quick) / 45 min (thorough) / 2 hours (exhaustive)
- Output includes 10-section structured sketch
- Multiple instances can run in parallel on different theorems
- Sketch can be expanded by Theorem Prover agent

---

**Now begin the Proof Sketcher protocol for the theorem(s) provided by the user.**
