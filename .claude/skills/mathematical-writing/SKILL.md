---
name: mathematical-writing
description: Write and publish mathematical documentation following Fragile framework standards. Use when drafting theorems/proofs, formatting mathematical content, ensuring LaTeX/MyST compliance, or preparing documents for Jupyter Book publication.
---

# Mathematical Writing Skill

## Purpose

Complete workflow for writing, formatting, and publishing mathematical documentation that meets Fragile framework standards and Jupyter Book requirements.

**Input**: Mathematical ideas, rough drafts, or existing documents needing formatting
**Output**: Publication-ready MyST markdown documents with proper formatting
**Pipeline**: Draft → Format → Dual-Review → Build → Publish

---

## Framework Standards

The Fragile project maintains **top-tier journal standards** for mathematical documentation:

### Rigor Requirements
- Every claim must have a complete proof or explicit reference
- All definitions must be unambiguous and mathematically precise
- Axioms and assumptions must be stated explicitly
- Proofs should include detailed step-by-step derivations
- Physical intuition should be provided in admonitions

### Format Requirements
- **LaTeX Math**: Use `$` for inline, `$$` for display math
- **Critical Spacing**: Always include exactly ONE blank line before opening `$$` blocks
- **Jupyter Book Directives**: Use `{prf:definition}`, `{prf:theorem}`, `{prf:proof}`, etc.
- **Labels**: All directives must have `:label:` for cross-referencing
- **Cross-references**: Use `{prf:ref}` for internal links

### Notation Consistency
All mathematical notation must match `docs/glossary.md` conventions:
- Greek letters (γ, β, σ, τ, etc.)
- Calligraphic letters (𝒳, 𝒴, 𝒮, etc.)
- Common operators (d_𝒳, φ, ψ, Ψ)

---

## Complete Workflow

### Stage 1: Drafting Content

**Purpose**: Create initial mathematical content with proper structure

#### Step 1.1: Consult the Mathematical Index

**ALWAYS START HERE** before writing any new content:

```bash
# Check if related content already exists
cat docs/glossary.md | grep -i "your-topic"

# Search by tags
cat docs/glossary.md | grep "tag:cloning"
cat docs/glossary.md | grep "tag:langevin"

# Find related definitions/theorems
cat docs/glossary.md | grep "thm-" | grep "convergence"
```

**Why important:**
- Avoid duplicate definitions
- Ensure notation consistency
- Identify dependencies
- Navigate to source documents

#### Step 1.2: Draft Content

Create content using proper MyST directives:

```markdown
:::{prf:definition} Euclidean Gas
:label: obj-euclidean-gas

A **Euclidean Gas** is a stochastic particle system...
:::

:::{prf:theorem} Main Convergence Result
:label: thm-euclidean-convergence

Under the stated axioms...
:::

:::{prf:proof}
We proceed in three steps...
:::
```

**Key points:**
- Use lowercase kebab-case for labels
- Include detailed proofs (not just sketches)
- Add physical intuition in `{note}` or `{tip}` admonitions
- Reference framework axioms explicitly

#### Step 1.3: Follow Template Structure

**For Definitions:**
```markdown
:::{prf:definition} Name
:label: obj-label-here

**Informal Description**: [Intuitive explanation]

**Formal Definition**: [Precise mathematical statement]

**Notation**: [Symbols introduced]

**Properties**: [Key properties if applicable]
:::
```

**For Theorems:**
```markdown
:::{prf:theorem} Name
:label: thm-label-here

**Statement**: [Precise theorem statement]

**Hypotheses**:
- [Assumption 1]
- [Assumption 2]

**Conclusion**: [What is proven]
:::

:::{prf:proof}
[Complete proof with numbered steps]

1. [Step 1 with justification]
2. [Step 2 with justification]
   ...
:::
```

---

### Stage 2: Formatting

**Purpose**: Ensure LaTeX/MyST compliance and notation consistency

#### Step 2.1: Fix Math Formatting

Run formatting tools in sequence:

```bash
# 1. Convert Unicode to LaTeX
python src/tools/convert_unicode_math.py docs/source/1_euclidean_gas/your_document.md

# 2. Convert backticks to dollar signs
python src/tools/convert_backticks_to_math.py docs/source/1_euclidean_gas/your_document.md

# 3. Fix spacing before $$ blocks (CRITICAL)
python src/tools/fix_math_formatting.py docs/source/1_euclidean_gas/your_document.md

# 4. Handle complex subscripts
python src/tools/fix_complex_subscripts.py docs/source/1_euclidean_gas/your_document.md

# 5. Comprehensive formatting pass
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md
```

**What each tool does:**
- `convert_unicode_math.py`: α → \alpha, ∇ → \nabla
- `convert_backticks_to_math.py`: `x` → $x$
- `fix_math_formatting.py`: Ensures blank line before `$$`
- `fix_complex_subscripts.py`: Handles nested subscripts
- `format_math_blocks.py`: All formatting fixes in one pass

#### Step 2.2: Verify Jupyter Book Compliance

Check directive syntax:

```bash
# Test build locally
make build-docs

# Check for errors in output
# Look for "WARNING" or "ERROR" messages
```

**Common issues:**
- Missing colons in directive markers (`::: not ::`)
- Wrong number of colons (4 instead of 3)
- Missing labels
- Incorrect label prefixes (use `obj-`, `thm-`, `lem-`, `axiom-`)

---

### Stage 3: Dual-Review Validation

**Purpose**: Critical review for rigor, consistency, and correctness

**MANDATORY**: Before finalizing any mathematical content, submit for dual-review.

#### Step 3.1: Prepare Review Request

Identify focus areas:
- New theorems/proofs to review
- Definitions to check for consistency
- Notation to verify against framework

#### Step 3.2: Submit to Both Reviewers

**Use IDENTICAL prompts** for both reviewers:

```
Review this mathematical document for:
1. Rigor: Are proofs complete and rigorous?
2. Consistency: Does notation match docs/glossary.md?
3. Correctness: Are all claims mathematically sound?
4. Clarity: Is the presentation clear and well-organized?

Document: [paste relevant sections]

Focus areas:
- Theorem 3.1 proof (lines 450-600)
- Definition 2.3 consistency with framework
- Notation for cloning operator
```

**Submit to:**
1. **Gemini 2.5 Pro** via `mcp__gemini-cli__ask-gemini` (model: "gemini-2.5-pro")
2. **Codex** via `mcp__codex__codex`

Run in parallel (single message with two tool calls).

#### Step 3.3: Compare and Synthesize Feedback

**Critical evaluation protocol:**

1. **Consensus Issues** (both agree): High confidence → prioritize
2. **Contradictions** (reviewers disagree): Investigate manually
3. **Unique Issues** (only one identifies): Verify against framework docs

**Always verify claims** by checking `docs/glossary.md` and source documents before accepting feedback.

#### Step 3.4: Implement Fixes

Address feedback systematically:
1. Start with CRITICAL issues (break mathematical correctness)
2. Then MAJOR issues (weaken claims or clarity)
3. Finally MINOR issues (improve presentation)

**After fixes**: Re-run dual-review to verify resolution.

---

### Stage 4: Building and Publishing

**Purpose**: Generate publication-ready HTML/PDF documentation

#### Step 4.1: Convert Mermaid Blocks (if needed)

If document contains mermaid diagrams:

```bash
# Manually convert (usually automatic during build)
python src/tools/convert_mermaid_blocks.py docs/source/1_euclidean_gas/your_document.md --in-place
```

Converts:
- ` ````mermaid ` → ` ```{mermaid} `
- `:::mermaid` → ` ```{mermaid} `

#### Step 4.2: Build Documentation

```bash
# Build Jupyter Book documentation
make build-docs

# Check build output for errors
# Look for "WARNING" or "ERROR" messages
```

**Common build errors:**
- Math blocks without blank line before `$$`
- Invalid directive syntax
- Broken cross-references
- Missing labels

#### Step 4.3: Serve and Preview

```bash
# Serve locally
make serve-docs

# Open browser to http://localhost:8000
```

**Check for:**
- Correct math rendering
- Working cross-references (links should be blue and clickable)
- Proper theorem/definition numbering
- Admonitions render correctly

#### Step 4.4: Final Quality Check

**Before committing:**
- [ ] All math renders correctly
- [ ] All cross-references work
- [ ] Notation matches glossary
- [ ] Proofs are complete
- [ ] Build has no warnings/errors
- [ ] Dual-review feedback addressed

---

## Mathematical Notation Reference

### Greek Letters (Code → LaTeX)

```python
# In Python code
gamma = 0.5        # Friction coefficient
beta = 2.0         # Exploitation weight
sigma = 0.1        # Perturbation noise
tau = 0.01         # Time step
epsilon_F = 0.05   # Regularization
```

```markdown
<!-- In markdown -->
$\gamma$, $\beta$, $\sigma$, $\tau$, $\epsilon_F$
```

### Common Operators

| Concept | Code | LaTeX | Meaning |
|---------|------|-------|---------|
| State space | X | `\mathcal{X}` | State space |
| Algorithmic space | Y | `\mathcal{Y}` | Algorithmic space |
| Swarm config | S | `\mathcal{S}` | Swarm configuration |
| Alive set | A | `\mathcal{A}` | Alive walkers |
| Metric | d_X | `d_\mathcal{X}` | Metric on state space |
| Projection | phi | `\phi` | Projection map |
| Squashing | psi | `\psi` | Squashing map |
| Operator | Psi_clone | `\Psi_{\text{clone}}` | Cloning operator |

### Subscript Conventions

```latex
<!-- Single indices -->
$x_i$, $v_i$, $w_i$

<!-- Algorithmic space -->
$x_i^{\mathcal{Y}}$

<!-- Time indices -->
$x_i^{(t)}$

<!-- Multiple subscripts (use braces) -->
$x_{i,j}$, $\sigma_{ij}$
```

---

## Common Writing Tasks

### Task 1: Add New Definition

**Workflow:**
```bash
# 1. Check if it exists
cat docs/glossary.md | grep -i "your-concept"

# 2. Draft definition with proper directive
vim docs/source/1_euclidean_gas/your_document.md

# 3. Format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md

# 4. Dual-review
# (Submit to Gemini + Codex)

# 5. Build and verify
make build-docs
```

**Template:**
```markdown
:::{prf:definition} Your Concept
:label: obj-your-concept

**Informal Description**: [Intuition]

**Formal Definition**: [Precise mathematical statement]

$$
\text{mathematical expression}
$$

**Notation**: [Symbols introduced]
:::
```

---

### Task 2: Write New Theorem with Proof

**Workflow:**
```bash
# 1. Check dependencies exist
cat docs/glossary.md | grep "thm-" | grep "related-topic"

# 2. Draft theorem + proof
vim docs/source/1_euclidean_gas/your_document.md

# 3. Format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md

# 4. Dual-review (MANDATORY for theorems)
# (Submit to Gemini + Codex with identical prompt)

# 5. Build and verify
make build-docs
```

**Template:**
```markdown
:::{prf:theorem} Your Result
:label: thm-your-result

**Statement**: [Precise theorem statement]

**Hypotheses**:
- [Assumption 1]
- [Assumption 2]

**Conclusion**: [What is proven]
:::

:::{prf:proof}
We proceed in three steps.

**Step 1**: [Establish foundation]

We have

$$
\text{equation with justification}
$$

**Step 2**: [Main derivation]

By {prf:ref}`lem-helper-lemma`, we obtain

$$
\text{key result}
$$

**Step 3**: [Conclude]

Combining the above yields the desired result.
:::
```

---

### Task 3: Fix Formatting Issues

**Common issues and fixes:**

#### Issue: Math not rendering

**Cause**: Missing blank line before `$$` block

**Fix:**
```bash
python src/tools/fix_math_formatting.py docs/source/1_euclidean_gas/your_document.md
```

**Manual fix:**
```markdown
<!-- WRONG -->
We have the equation:
$$
x = y
$$

<!-- CORRECT -->
We have the equation:

$$
x = y
$$
```

#### Issue: Broken cross-reference

**Cause**: Label doesn't exist or wrong syntax

**Fix:**
```bash
# Find the correct label
cat docs/glossary.md | grep "your-concept"

# Use correct reference syntax
{prf:ref}`thm-correct-label`
```

#### Issue: Unicode math not converting

**Cause**: Special characters in document

**Fix:**
```bash
python src/tools/convert_unicode_math.py docs/source/1_euclidean_gas/your_document.md
```

---

## Integration with Other Skills

### From Extract-and-Refine

After extracting entities, you may need to write surrounding narrative:

```bash
# Extracted entities in refined_data/
ls docs/source/1_euclidean_gas/02_euclidean_gas/refined_data/

# Write narrative documentation around extracted entities
vim docs/source/1_euclidean_gas/02_euclidean_gas.md
```

### To Proof-Validation

After writing theorem statements, develop proofs:

```
# Write theorem statement (mathematical-writing)
:::{prf:theorem} New Result
:label: thm-new-result
...
:::

# Then use proof-validation skill:
Load proof-sketcher agent.
Sketch: thm-new-result from docs/source/1_euclidean_gas/document.md
```

### To Framework-Consistency

After writing content, verify consistency:

```
# Draft content (mathematical-writing)
vim docs/source/1_euclidean_gas/document.md

# Verify consistency (framework-consistency skill)
# Check notation, verify axiom usage, validate claims
```

---

## Best Practices

### 1. Always Consult Glossary First

Before writing anything new:
```bash
cat docs/glossary.md | grep -i "your-topic"
```

Avoids:
- Duplicate definitions
- Notation inconsistencies
- Missing dependencies

### 2. Use Formatting Tools

Always run formatting tools before dual-review:
```bash
python src/tools/format_math_blocks.py your_document.md
```

Ensures consistent style.

### 3. Dual-Review All Theorems

**Never skip dual-review** for theorems/proofs:
- Gemini 2.5 Pro: Deep mathematical analysis
- Codex: Independent verification
- Compare results: Identify hallucinations

### 4. Build Frequently

Check rendering after major changes:
```bash
make build-docs && make serve-docs
```

Catch formatting issues early.

### 5. Write Complete Proofs

Don't write "proof sketch" - write full derivations:
- Show all steps
- Justify every claim
- Reference axioms/lemmas explicitly

---

## Output Locations

All mathematical documents should be placed in:

```
docs/source/
├── 1_euclidean_gas/
│   ├── 01_fragile_gas_framework.md    # Core axioms
│   ├── 02_euclidean_gas.md             # Euclidean Gas spec
│   ├── 03_cloning.md                   # Cloning operator
│   ├── 04_convergence.md               # Convergence proofs
│   └── ...
├── 2_geometric_gas/
│   └── ...
└── glossary.md                         # Auto-generated index
```

---

## Quality Checklist

Before finalizing any document:

- [ ] Consulted `docs/glossary.md` for existing definitions
- [ ] Used proper MyST directive syntax
- [ ] All math uses `$` / `$$` (not backticks)
- [ ] Blank line before all `$$` blocks
- [ ] All directives have `:label:` fields
- [ ] Labels use lowercase kebab-case
- [ ] Cross-references use `{prf:ref}`
- [ ] Notation matches glossary conventions
- [ ] Proofs are complete with detailed steps
- [ ] Submitted to dual-review (Gemini + Codex)
- [ ] Addressed all CRITICAL/MAJOR feedback
- [ ] Verified feedback against framework docs
- [ ] Ran formatting tools
- [ ] Built documentation successfully (no warnings)
- [ ] Previewed rendering locally

---

## Related Documentation

- **CLAUDE.md**: Complete mathematical writing standards
- **GEMINI.md**: Gemini review protocol
- **docs/glossary.md**: Mathematical notation reference
- **Formatting Tools**: `src/tools/` directory
- **Extract-and-Refine Skill**: Entity extraction workflow
- **Proof-Validation Skill**: Proof development workflow
- **Framework-Consistency Skill**: Consistency verification

---

## Version History

- **v1.0.0** (2025-10-28): Initial mathematical-writing skill
  - Documented complete writing workflow
  - Added formatting tool usage
  - Integrated dual-review protocol
  - Provided notation reference and templates

---

**Next**: See [QUICKSTART.md](./QUICKSTART.md) for copy-paste commands.
