# Claude Code Agents for Fragile Framework

This directory contains specialized agents for autonomous tasks in the Fragile mathematical framework.

## Available Agents

### 📊 Math Reviewer (`math-reviewer.md`)

**Purpose**: Autonomous dual-review system for mathematical documents using Gemini 2.5 Pro + Codex

**Capabilities**:
- Strategic extraction from massive documents (>400KB)
- Parallel dual-review orchestration with identical prompts
- Critical comparative analysis (not just reporting)
- Framework cross-validation against `docs/glossary.md`
- Evidence-based judgment when reviewers disagree
- Comprehensive structured reporting
- Automatic file output to `reviewer/review_{timestamp}_{filename}.md`

**Parallelizable**: ✅ Yes (run multiple reviews simultaneously)

**Independent**: ✅ Yes (doesn't rely on slash commands or other agents)

---

### 📐 Proof Sketcher (`proof-sketcher.md`)

**Purpose**: Autonomous proof strategy generation for mathematical theorems using Gemini 2.5 Pro + GPT-5 Pro

**Capabilities**:
- Strategic theorem extraction from documents
- Parallel dual proof strategy generation
- Critical proof approach comparison (direct, constructive, contradiction, coupling, Lyapunov, etc.)
- Framework dependency verification against `docs/glossary.md`
- Proof sketch synthesis from best elements of both strategists
- Automatic file output to `sketcher/sketch_{timestamp}_proof_{filename}.md`

**Parallelizable**: ✅ Yes (run multiple proof sketches simultaneously)

**Independent**: ✅ Yes (doesn't rely on slash commands or other agents)

**Key Distinction from Math Reviewer**:
- **Math Reviewer**: Finds errors in *existing* proofs (quality control)
- **Proof Sketcher**: Creates *new* proof strategies (proof development)

---

### 🔬 Theorem Prover (`theorem-prover.md`)

**Purpose**: Autonomous proof expansion from sketches to publication-ready complete proofs using Gemini 2.5 Pro + GPT-5 Pro

**Capabilities**:
- Strategic sketch analysis and expansion roadmap generation
- Parallel dual proof expansion with identical prompts to both AIs
- Step-by-step rigorous expansion (epsilon-delta, measure theory, constants)
- Critical comparison and synthesis of expansions (13-point rigor checklist)
- Framework dependency verification and constant tracking
- Edge case handling (k=1, N→∞, boundary, degeneracies)
- Counterexample generation for hypothesis necessity
- Publication readiness assessment (Annals of Mathematics standard)
- Automatic file output to `proofs/proof_{timestamp}_{theorem_label}.md`

**Parallelizable**: ✅ Yes (run multiple proof expansions simultaneously)

**Independent**: ✅ Yes (doesn't rely on slash commands or other agents)

**Rigor Standard**: Annals of Mathematics (top-tier journal level)

**Key Distinctions**:
- **Math Reviewer**: Quality control on *existing* proofs (find errors)
- **Proof Sketcher**: Strategy generation (3-7 high-level steps)
- **Theorem Prover**: Complete expansion (every detail, ~500-2000 lines per proof)

---

### 🧮 Math Verifier (`math-verifier.md`)

**Purpose**: Autonomous symbolic validation of algebraic manipulations using sympy with Gemini 2.5 Pro + GPT-5 Pro

**Capabilities**:
- Strategic extraction of algebraic claims from documents
- Automated validation category detection (variance, logarithms, quadratic forms, etc.)
- Parallel dual AI code generation for validation scripts
- Framework symbol integration from `docs/glossary.md`
- Sympy script execution and result collection
- Pytest-compatible test suite generation
- Document annotation with verification markers (✓ verified / ⚠️ semantic)
- Automatic file output to `src/proofs/{doc_name}/{theorem_label}.py` and `verifier/verification_{timestamp}_{filename}.md`

**Parallelizable**: ✅ Yes (run multiple validations simultaneously)

**Independent**: ✅ Yes (doesn't rely on slash commands or other agents)

**Validation Engine**: sympy (Python symbolic mathematics library)

**Key Distinctions**:
- **Math Reviewer**: Semantic validation (proof logic, assumptions, completeness)
- **Math Verifier**: Algebraic validation (equation correctness, derivation steps)
- **Proof Sketcher**: Strategy generation (proof approach)
- **Theorem Prover**: Proof expansion (complete details)

**Complementary Role**: Math Verifier validates algebra BEFORE semantic review, catching computational errors early and reducing reviewer cognitive load on mechanical steps.

---

## How to Use Math Reviewer

### Method 1: Via Task Tool (Programmatic)

```python
from antml.tools import Task

Task(
    description="Review geometric gas C^∞ regularity",
    subagent_type="general-purpose",
    prompt="""
    Load instructions from: .claude/agents/math-reviewer.md

    Review the mathematical document at:
    docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md

    Depth: thorough
    Focus on: Non-circular density bound, k-uniformity claims, statistical equivalence
    """
)
```

### Method 2: Direct Invocation (Simpler)

Just ask Claude:
```
Load the math-reviewer agent and review:
@docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md

Focus on the non-circular density bound proof and k-uniformity claims.
```

### Method 3: Parallel Reviews

Launch multiple reviews simultaneously:
```
Run 3 math-reviewer agents in parallel:
1. Review @docs/source/1_euclidean_gas/03_cloning.md (focus: Keystone Principle)
2. Review @docs/source/2_geometric_gas/19_geometric_gas_cinf_regularity_simplified.md
3. Review @docs/source/3_brascamp_lieb/eigenvalue_gap_complete_proof.md (depth: exhaustive)
```

---

## Input Formats

The agent accepts these formats:

### Basic
```
Review: docs/source/path/to/document.md
```

### With Focus
```
Review: docs/source/path/to/document.md
Focus on: Section 2.3 non-circularity, Appendix A combinatorics
```

### With Depth
```
Review: docs/source/path/to/document.md
Depth: exhaustive
Focus: All LSI constants and spectral gap bounds
```

**Depth Levels**:
- `quick`: Abstract + main theorems only (~10 min)
- `thorough`: Key sections including proofs (~30-45 min, default)
- `exhaustive`: Complete document analysis (~1-2 hours)

---

## Output Format

The agent produces a comprehensive report with:

1. **Comparison Overview** (consensus vs contradictions)
2. **Issue Summary Table** (compact view with Gemini/Codex/Claude verdicts)
3. **Issue Analysis Table** (detailed sortable by severity)
4. **Detailed Issue-by-Issue Analysis** with:
   - Both reviewers' positions (quoted verbatim)
   - Agent's evidence-based judgment
   - Framework cross-validation results
   - Proposed fixes with rationale
5. **Implementation Checklist** (prioritized by severity)
6. **Contradictions** requiring user decision (3 perspectives)
7. **Final Verdict** with independent assessment

**File Output**:
- Automatically written to `reviewer/review_{timestamp}_{filename}.md` in document's directory
- Timestamp format: `YYYYMMDD_HHMM` (e.g., `20251024_1430`)
- Example: `docs/source/2_geometric_gas/reviewer/review_20251024_1430_20_geometric_gas_cinf_regularity_full.md`

Example output structure:
```
# Dual Review Summary for 20_geometric_gas_cinf_regularity_full.md

## Comparison Overview
- Consensus Issues: 4 (both reviewers agree)
- Gemini-Only Issues: 1
- Codex-Only Issues: 2
- Contradictions: 1 (reviewers disagree)

## Issue Summary Table
| # | Issue | Severity | Location | Gemini | Codex | Claude | Status |
|---|-------|----------|----------|--------|-------|--------|--------|
| 1 | Non-circular density | CRITICAL | §2.3.5, lines 450-465 | CRITICAL - Insufficient | CRITICAL - Not compact | ✅ Verified | ✗ Contradicts |

## Issue Analysis Table
| # | Issue | Severity | Gemini | Codex | Claude | Verification |
|---|-------|----------|--------|-------|--------|--------------|
| 1 | Non-circular density | CRITICAL | ... | ... | ✅ Verified | ...

[Full detailed analysis follows...]

✅ Review written to: docs/source/.../reviewer/review_20251024_1430_20_geometric_gas_cinf_regularity_full.md
```

---

## How to Use Proof Sketcher

### Method 1: Single Theorem by Label

```
Load proof-sketcher agent.

Sketch proof for theorem: thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
Depth: thorough
```

### Method 2: Document with Focus

```
Load proof-sketcher.

Sketch proofs for: docs/source/1_euclidean_gas/06_convergence.md
Focus on: Foster-Lyapunov main theorem and drift lemmas
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

### Input Parameters

- **file_path** (required): Path to document containing theorems
- **theorems** (optional): Specific theorem labels (comma-separated)
- **focus_areas** (optional): Topic descriptions
- **depth** (optional): `quick` | `thorough` (default) | `exhaustive`

### Output Format

**File Location**: `sketcher/sketch_{YYYYMMDD_HHMM}_proof_{document_name}.md`

**Sketch Structure** (10 sections):
1. Theorem Statement (exact copy from source)
2. Proof Strategy Comparison (Gemini vs GPT-5 vs Claude synthesis)
3. Framework Dependencies (verified against glossary)
4. Detailed Proof Sketch (step-by-step with justifications)
5. Technical Deep Dives (1-3 hardest challenges)
6. Proof Validation Checklist
7. Alternative Approaches (not chosen)
8. Open Questions & Future Work
9. Expansion Roadmap (time estimates)
10. Cross-References

Example file: `docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md`

---

## How to Use Theorem Prover

### Method 1: Expand from Sketch File

```
Load theorem-prover agent.

Expand proof sketch:
docs/source/1_euclidean_gas/sketcher/sketch_20251024_1530_proof_09_kl_convergence.md
```

### Method 2: By Theorem Label (Auto-Find Sketch)

```
Load theorem-prover.

Expand proof for theorem: thm-kl-convergence-euclidean
Document: docs/source/1_euclidean_gas/09_kl_convergence.md
```

Agent will automatically find the most recent sketch for this theorem.

### Method 3: With Focus Areas

```
Load theorem-prover.

Expand proof sketch: sketcher/sketch_20251024_1530_proof_theorem.md
Focus on:
- Step 4: Add complete epsilon-delta for all limits
- Step 5: Verify all Fubini conditions explicitly
- All steps: Track all constants with explicit formulas
```

### Method 4: Expand Specific Steps Only

```
Load theorem-prover.

Expand steps 4-5 from sketch:
sketcher/sketch_20251024_1530_proof_theorem.md

Focus on: Add complete Fisher information derivation
Keep other steps as sketched
```

### Input Parameters

- **sketch_file** (optional): Path to specific sketch file
- **theorem_label** (optional): Theorem label to find sketch for
- **document_path** (optional): Document containing theorem
- **focus_areas** (optional): Specific technical elements to emphasize
- **depth** (optional): `standard` (default, ~2-3 hours) | `maximum` (every epsilon, ~4-6 hours)

### Output Format

**File Location**: `proofs/proof_{YYYYMMDD_HHMM}_{theorem_label}.md`

**Complete Proof Structure** (9 sections):
1. **Theorem Statement** (exact from source with label)
2. **Proof Expansion Comparison** (Gemini vs GPT-5 vs Claude synthesis with rigor scores)
3. **Framework Dependencies** (verified tables against glossary.md)
4. **Complete Rigorous Proof** (every detail expanded, ~500-2000 lines)
5. **Verification Checklist** (epsilon-delta, measure theory, constants, edge cases)
6. **Edge Cases Handled** (k=1, N→∞, boundary, degeneracies)
7. **Counterexamples for Necessity** (showing all hypotheses required)
8. **Publication Readiness Assessment** (numerical scores + verdict)
9. **Cross-References** (framework dependencies and related results)

**Example Output**:
- Length: ~500-2000 lines (depends on theorem complexity)
- Rigor: 8-10/10 (publication standard)
- Time: ~2-4 hours for standard depth

Example file: `docs/source/1_euclidean_gas/proofs/proof_20251024_1630_thm_kl_convergence_euclidean.md`

---

## How to Use Math Verifier

### Method 1: Validate Document
```
Load math-verifier agent.

Validate algebraic manipulations in:
docs/source/1_euclidean_gas/03_cloning.md
```

### Method 2: Validate Specific Theorem
```
Load math-verifier.

Validate theorem: thm-keystone-lemma
Document: docs/source/1_euclidean_gas/03_cloning.md
```

### Method 3: Validate Proof Sketch
```
Load math-verifier.

Validate sketch:
sketcher/sketch_20251024_1530_proof_thm_keystone.md
```

### Method 4: Validate Complete Proof
```
Load math-verifier.

Validate proof:
proofs/proof_20251024_1630_thm_keystone_lemma.md
```

### Method 5: With Depth and Focus
```
Load math-verifier.

Validate: docs/source/1_euclidean_gas/03_cloning.md
Depth: exhaustive
Focus: Variance decompositions, logarithmic bounds, signal propagation
```

### Input Parameters
- **file_path** (required): Path to document/sketch/proof to validate
- **theorem_label** (optional): Specific theorem to focus on
- **focus_areas** (optional): Algebraic categories (variance, logarithms, etc.)
- **depth** (optional): `quick` | `thorough` (default) | `exhaustive`

### Output Format

**Validation Scripts**: `src/proofs/{doc_name}/{theorem_label}.py`

**Script Structure**:
```python
"""
Symbolic Validation for {Theorem Title}
Source: {file_path}
Generated: {timestamp}
"""
from sympy import symbols, simplify, expand

def test_{theorem_label}_step1():
    # Validate algebraic step 1
    ...

def test_{theorem_label}_step2():
    # Validate algebraic step 2
    ...

if __name__ == "__main__":
    run_all_validations()
```

**Verification Report**: `verifier/verification_{YYYYMMDD_HHMM}_{filename}.md`

**Report Structure**:
1. Validation Overview (claims found, validated, passed/failed)
2. Category Breakdown (variance, logarithms, quadratic forms, etc.)
3. Detailed Results (step-by-step with ✅ PASSED / ❌ FAILED)
4. Code Generation Comparison (Gemini vs GPT-5)
5. Framework Integration (symbols from glossary.md)
6. Validation Failures (action required)
7. Semantic Steps (not validated - require Math Reviewer)

**Example Output**:
- Scripts: `src/proofs/03_cloning/lem_variance_to_mean_separation.py`
- Report: `docs/source/1_euclidean_gas/verifier/verification_20251024_1430_03_cloning.md`
- Pass rate: 15/16 tests passed (93.75%)

---

## Best Practices

### When to Use Math Reviewer

✅ **Use for**:
- New mathematical documents before publication
- Existing documents with suspected errors
- Critical proofs requiring validation
- Documents >1000 lines needing strategic analysis
- Cross-checking complex logical chains

❌ **Don't use for**:
- Trivial edits or typo fixes
- Style/formatting issues (use formatting tools instead)
- Code reviews (different agent needed)
- Documents you want Claude to edit directly (use /dual_review command instead)

### Tips for Best Results

1. **Be specific about focus areas**: The more specific, the deeper the analysis
   ```
   Focus on: Lines 450-650 (density bound proof), Lemma 3.2 (telescoping identity)
   ```

2. **Choose appropriate depth**:
   - `quick`: For initial sanity check
   - `thorough`: For standard review (recommended)
   - `exhaustive`: For final pre-publication review

3. **Review in stages** for very large documents:
   ```
   Stage 1: Review Part I (Foundations) - thorough
   Stage 2: Review Part II (Main Results) - exhaustive
   Stage 3: Review Appendices - quick
   ```

4. **Use parallel execution** for efficiency:
   - Math Reviewer instances are independent
   - Each produces complete report
   - Compare reports if reviewing same document from different angles

---

### When to Use Proof Sketcher

✅ **Use for**:
- New theorems that need proofs
- Existing theorems with incomplete/unclear proofs
- Theorems where proof strategy is uncertain
- Complex theorems requiring careful planning
- Theorems with multiple possible approaches

❌ **Don't use for**:
- Theorems with complete rigorous proofs (use Math Reviewer instead)
- Trivial lemmas with obvious proofs (just write the proof directly)
- Theorems outside Fragile framework (agent won't have context)

### Tips for Proof Sketcher

1. **Be specific about focus**:
   ```
   Focus on: Verify all constants are N-uniform, track k-dependence
   ```

2. **Choose appropriate depth**:
   - `quick`: Initial exploration, sanity check (~15 min)
   - `thorough`: Standard workflow (~45 min, recommended)
   - `exhaustive`: All theorems in document (~2 hours)

3. **Iterate on sketches**:
   - First pass: Get overall strategy
   - Prove missing lemmas
   - Second pass: Re-sketch with new lemmas available
   - Expand to full proof

4. **Use parallel execution** for multiple theorems:
   - Proof Sketcher instances are independent
   - Each produces complete sketch
   - Can sketch different documents simultaneously

---

### When to Use Theorem Prover

✅ **Use for**:
- Expanding proof sketches to publication-ready complete proofs
- Converting high-level strategies (3-7 steps) into detailed proofs (500-2000 lines)
- Ensuring Annals of Mathematics rigor standard
- Completing proofs requiring extensive epsilon-delta arguments
- Verifying all measure theory operations are justified
- Handling all edge cases systematically (k=1, N→∞, boundary)
- Generating counterexamples for hypothesis necessity
- Final proof preparation before publication

❌ **Don't use for**:
- Creating proof strategies from scratch (use Proof Sketcher first)
- Theorems with trivial proofs (expand manually)
- Quality control on existing complete proofs (use Math Reviewer instead)
- Sketches that haven't been validated (validate strategy first)

### Tips for Theorem Prover

1. **Always sketch first**: Theorem Prover requires a validated proof strategy
   ```
   # Recommended workflow:
   1. Proof Sketcher → Generate strategy
   2. Math Reviewer → Validate sketch (optional)
   3. Theorem Prover → Expand to complete proof
   4. Math Reviewer → Final validation
   ```

2. **Handle missing lemmas before expansion**:
   - Agent detects missing lemmas during sketch analysis
   - Prove dependencies first, then expand main theorem
   - Or use recursive expansion mode (agent proves lemmas automatically)

3. **Choose appropriate depth**:
   - `standard`: Normal Annals level (~2-3 hours, recommended)
   - `maximum`: Absolute maximum detail (~4-6 hours, for most critical theorems)

4. **Use focus areas for specific technical needs**:
   ```
   Focus on:
   - Step 4: Complete all epsilon-delta arguments
   - All steps: Track all constants explicitly
   - Edge case: k=1 single walker behavior
   ```

5. **Iterate if needed**:
   - First expansion at standard rigor → assessment 7-8/10
   - Address remaining tasks → re-expand specific steps
   - Final expansion → assessment 9-10/10

6. **Use parallel execution** for multiple theorems:
   - Theorem Prover instances are independent
   - Each produces complete proof
   - Can expand different theorems simultaneously

---

### When to Use Math Verifier

✅ **Use for**:
- Validating algebraic manipulations in proofs (equations, derivations, identities)
- Catching computational errors before semantic review
- Generating executable certificates for algebraic correctness
- Creating pytest test suites for continuous validation
- Validating proof sketches (early error detection)
- Validating complete proofs (comprehensive validation)
- Documents with complex algebra (variance decompositions, log identities, quadratic forms)
- Before Math Reviewer (reduce cognitive load on mechanical steps)

❌ **Don't use for**:
- Semantic reasoning validation (use Math Reviewer instead)
- Measure-theoretic arguments (not algebraic)
- Topological claims (compactness, continuity)
- Proof structure or logic (use Math Reviewer)
- Axiom dependency verification (use Math Reviewer)

### Tips for Math Verifier

1. **Run early in pipeline**: Validate algebra before semantic review
   ```
   # Recommended: Dual validation workflow
   Proof Sketcher → Math Verifier → Theorem Prover → Math Verifier → Math Reviewer
   ```

2. **Focus on error-prone categories**:
   ```
   Focus: Variance decompositions, logarithmic identities, quadratic form expansions
   ```

3. **Use with existing documents**:
   ```
   # Validate document you're editing
   /math_verify docs/source/1_euclidean_gas/03_cloning.md
   ```

4. **Continuous validation** after edits:
   ```bash
   # Re-run validation scripts after document changes
   pytest src/proofs/03_cloning/
   ```

5. **Interpret results**:
   - ✅ **PASSED**: Algebra is correct (computational certificate)
   - ❌ **FAILED**: Algebra error found (review source document)
   - ⚠️ **SEMANTIC**: Not algebraic (requires Math Reviewer)

6. **Use parallel execution** for multiple documents:
   - Math Verifier instances are independent
   - Each produces validation scripts + report
   - Can validate different documents simultaneously

---

## Agent Guarantees

### Math Reviewer Guarantees

The Math Reviewer agent guarantees:

1. ✅ **Identical prompts** to both Gemini 2.5 Pro and Codex
2. ✅ **Evidence-based judgments** backed by framework verification
3. ✅ **Honest uncertainty** acknowledgment when verification impossible
4. ✅ **Actionable fixes** for every CRITICAL/MAJOR issue
5. ✅ **Template compliance** for parseable output
6. ✅ **Framework consistency** checks against glossary.md
7. ✅ **Independent critical analysis** (not passive reporting)

### Proof Sketcher Guarantees

The Proof Sketcher agent guarantees:

1. ✅ **Framework Consistency**: All dependencies verified in glossary.md
2. ✅ **No Circular Reasoning**: Proof steps don't assume conclusion
3. ✅ **Constant Tracking**: All mathematical constants defined and bounded
4. ✅ **Logical Completeness**: All parts of theorem statement addressed
5. ✅ **Actionable Steps**: Every step can be expanded to full rigorous proof
6. ✅ **Alternative Documentation**: Other approaches preserved for future use
7. ✅ **Independent Strategies**: Gemini + GPT-5 provide distinct perspectives

### Theorem Prover Guarantees

The Theorem Prover agent guarantees:

1. ✅ **Annals of Mathematics Rigor**: Publication-ready proofs with complete details
2. ✅ **Complete Epsilon-Delta**: All limits proven with explicit ε-δ arguments
3. ✅ **Measure Theory Justified**: All operations verified (Fubini conditions, DCT, etc.)
4. ✅ **Explicit Constants**: All constants have formulas (no unjustified O(1))
5. ✅ **Edge Cases Handled**: k=1, N→∞, boundary, degeneracies all addressed
6. ✅ **Counterexamples Provided**: Necessity of all hypotheses demonstrated
7. ✅ **Dual AI Synthesis**: Best elements from Gemini + GPT-5 combined
8. ✅ **Framework Verification**: All dependencies cross-checked against glossary.md
9. ✅ **Rigor Scoring**: 13-point checklist per step with numerical assessment
10. ✅ **Publication Assessment**: Objective readiness verdict with scores

### Math Verifier Guarantees

The Math Verifier agent guarantees:

1. ✅ **Algebraic Coverage**: Identify ≥90% of algebraic claims in document
2. ✅ **Dual AI Code Generation**: Validation code from both Gemini 2.5 Pro + GPT-5 Pro
3. ✅ **Executable Scripts**: All generated code runs without modification
4. ✅ **Framework Integration**: Symbols and constants from glossary.md
5. ✅ **Category Detection**: Automated classification (variance, log, quadratic, etc.)
6. ✅ **Pytest Compatibility**: Test suite for continuous validation
7. ✅ **Clear Separation**: Verified (✓) vs semantic (⚠️) steps documented
8. ✅ **Validation Execution**: All scripts executed with pass/fail results
9. ✅ **Comprehensive Reporting**: Category breakdown, code comparison, failures
10. ✅ **Continuous Validation**: Re-runnable scripts after document edits

---

## Troubleshooting

### Agent doesn't find document
- Verify file path is correct
- Use @ prefix for file references
- Check file exists: `ls -lh docs/source/path/to/file.md`

### Review takes too long
- Check depth setting (use `quick` for faster results)
- Reduce focus areas to specific sections
- Document might be very large (>5000 lines = ~1 hour for thorough)

### Gemini or Codex fails
- Agent will complete with single reviewer
- Report will note limitation
- Recommend re-running when service available

### Want to stop mid-review
- Math Reviewer runs as autonomous agent
- Cannot pause mid-execution
- Will complete and produce full report

---

## Maintenance

**Agent Version**: 1.0
**Last Updated**: 2025-10-24
**Compatible With**: Claude Code, Gemini 2.5 Pro API, Codex API

**Update History**:
- v1.0 (2025-10-24): Initial release with dual-review protocol

---

## Examples

### Example 1: Standard Review
```
Load math-reviewer agent.

Review: docs/source/2_geometric_gas/20_geometric_gas_cinf_regularity_full.md
Depth: thorough
```

**Expected Runtime**: 30-45 minutes
**Output**: ~5000-8000 word comprehensive report

### Example 2: Focused Quick Check
```
Load math-reviewer.

Review: docs/source/1_euclidean_gas/03_cloning.md
Depth: quick
Focus: Keystone Principle proof (§8), companion selection (§5.1)
```

**Expected Runtime**: 10-15 minutes
**Output**: ~2000-3000 word focused report

### Example 3: Pre-Publication Exhaustive Review
```
Load math-reviewer.

Review: docs/source/3_brascamp_lieb/eigenvalue_gap_complete_proof.md
Depth: exhaustive
Focus: All proofs, all lemmas, all computational bounds
```

**Expected Runtime**: 1-2 hours
**Output**: ~10000+ word complete analysis

---

## Integration with Workflow

### Recommended Review Workflow (Existing Proofs)

1. **Initial Draft** → Use `/dual_review` slash command (interactive)
2. **Revision 1** → Use Math Reviewer agent (thorough)
3. **Address Issues** → Implement fixes from agent report
4. **Final Check** → Use Math Reviewer agent (exhaustive)
5. **Publication** → Document passes with "READY" or "MINOR REVISIONS"

### Recommended Proof Development Workflow (New Theorems) - ENHANCED WITH SYMBOLIC VALIDATION

**Dual Validation Workflow** (RECOMMENDED - Most Rigorous):

1. **Strategy Generation** → Use Proof Sketcher agent (~45 min)
   - Output: `sketcher/sketch_{timestamp}_proof_{filename}.md`

2. **Algebraic Validation (Strategy)** → Use Math Verifier agent (~20 min)
   - Validates algebraic steps in proof strategy
   - Catches errors before expansion
   - Output: `src/proofs/{doc_name}/` validation scripts

3. **Proof Expansion** → Use Theorem Prover agent (~2-4 hours)
   - Expands to complete proof with all details
   - Can reference verified algebraic steps
   - Output: `proofs/proof_{timestamp}_{theorem_label}.md`

4. **Algebraic Validation (Proof)** → Use Math Verifier agent (~30 min)
   - Validates detailed algebraic expansions
   - Generates comprehensive pytest suite
   - Identifies any new algebraic errors

5. **Semantic Validation** → Use Math Reviewer agent (exhaustive)
   - Focuses on proof logic and structure
   - Algebraic steps already verified
   - Reduced cognitive load on mechanical steps

6. **Publication** → Proof meets Annals of Mathematics standard

**Quick Validation Workflow** (Faster, Less Error Prevention):

1. Proof Sketcher (~45 min)
2. Theorem Prover (~2-4 hours)
3. Math Verifier (~30 min) - validates complete proof
4. Math Reviewer (~1 hour) - semantic validation
5. Publication

### Complete Pipeline with Symbolic Validation

**For New Theorems (Dual Validation - RECOMMENDED)**:
```
┌──────────────────┐
│ Proof Sketcher   │ ──→ Strategy (3-7 steps, ~45 min)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Math Verifier   │ ──→ Validate strategy algebra (~20 min)
└────────┬─────────┘     Catches errors early ✓
         │                Output: validation scripts
         ▼
┌──────────────────┐
│ Theorem Prover   │ ──→ Expand to complete proof (~2-4 hours)
└────────┬─────────┘     Every detail, Annals standard
         │
         ▼
┌──────────────────┐
│  Math Verifier   │ ──→ Validate proof algebra (~30 min)
└────────┬─────────┘     Comprehensive pytest suite ✓
         │                Output: validation scripts + report
         ▼
┌──────────────────┐
│  Math Reviewer   │ ──→ Semantic validation (~1 hour)
└────────┬─────────┘     Proof logic, framework consistency
         │                Algebra already verified
         ▼
   Publication ✓
   - Mathematically rigorous
   - Algebraically verified (executable certificates)
   - Semantically validated (dual AI review)
```

**For Existing Documents (Quality Control)**:
```
┌──────────────────┐
│  Math Verifier   │ ──→ Validate algebra first (~30 min)
└────────┬─────────┘     Computational correctness
         │
         ▼
┌──────────────────┐
│  Math Reviewer   │ ──→ Semantic review (~45 min)
└────────┬─────────┘     Focus on logic, not algebra
         │
         ▼
   Revision & Re-validation
```

### Collaboration with Other Tools

- **After Math Reviewer**: Use formatting tools from `src/tools/`
- **Before Math Reviewer**: Use `/introduction` or `/clean_ai` commands
- **Parallel execution**: Run multiple agents simultaneously on different documents/theorems
- **After Theorem Prover**: Use `/clean_ai` to remove AI meta-commentary before publication

---

## Support

For issues or questions about these agents:

**Math Reviewer**:
1. Check this README
2. Read the agent definition: `.claude/agents/math-reviewer.md`
3. See quick start: `.claude/agents/math-reviewer-QUICKSTART.md`

**Proof Sketcher**:
1. Check this README
2. Read the agent definition: `.claude/agents/proof-sketcher.md`
3. See quick start: `.claude/agents/proof-sketcher-QUICKSTART.md`

**Theorem Prover**:
1. Check this README
2. Read the agent definition: `.claude/agents/theorem-prover.md`
3. See comprehensive docs: `.claude/agents/theorem-prover-README.md`
4. See quick start: `.claude/agents/theorem-prover-QUICKSTART.md`

**Math Verifier**:
1. Check this README
2. Read the agent definition: `.claude/agents/math-verifier.md`
3. See use cases: `SYMPY_USE_CASES.md`
4. Example scripts: `src/proofs/{doc_name}/`

**General Support**:
1. Consult CLAUDE.md § Mathematical Proofing and Documentation
2. Open issue at: https://github.com/anthropics/claude-code/issues

---

**Happy Proving! 📊🔍🔬**
