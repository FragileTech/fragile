# Mathematical Writing - Quick Start

Copy-paste ready commands for common mathematical writing tasks.

---

## Quick Lookup

### Check if concept already exists
```bash
# Search glossary for existing definitions
cat docs/glossary.md | grep -i "your-concept"

# Search by tags
cat docs/glossary.md | grep "tag:cloning"
cat docs/glossary.md | grep "tag:langevin"

# Find theorems by topic
cat docs/glossary.md | grep "thm-" | grep "convergence"
```

---

## New Definition

### 1. Check + Draft + Format + Review
```bash
# Check if exists
cat docs/glossary.md | grep -i "euclidean-gas"

# Edit document
vim docs/source/1_euclidean_gas/02_euclidean_gas.md

# Format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/02_euclidean_gas.md

# Build and verify
make build-docs
```

### Template
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

## New Theorem + Proof

### 1. Check Dependencies + Draft + Format + Dual-Review + Build
```bash
# Check dependencies
cat docs/glossary.md | grep "thm-" | grep "related-topic"

# Edit document
vim docs/source/1_euclidean_gas/03_cloning.md

# Format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/03_cloning.md

# Dual-review (in Claude Code)
# Submit IDENTICAL prompt to Gemini + Codex (see template below)

# Build
make build-docs
```

### Dual-Review Prompt (use for BOTH Gemini + Codex)
```
Review this theorem and proof for:
1. Mathematical rigor (are all claims proven?)
2. Consistency with framework (check docs/glossary.md)
3. Correctness (are derivations sound?)
4. Clarity (is proof well-organized?)

[Paste theorem + proof here]

Focus areas:
- Step 3 derivation (potential gap)
- Notation consistency with framework
- Completeness of proof
```

### Template
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

## Fix Formatting Issues

### All formatting fixes in one command
```bash
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md
```

### Individual fixes (if needed)
```bash
# Fix spacing before $$ blocks (CRITICAL)
python src/tools/fix_math_formatting.py docs/source/1_euclidean_gas/your_document.md

# Convert Unicode to LaTeX
python src/tools/convert_unicode_math.py docs/source/1_euclidean_gas/your_document.md

# Convert backticks to dollar signs
python src/tools/convert_backticks_to_math.py docs/source/1_euclidean_gas/your_document.md

# Fix complex subscripts
python src/tools/fix_complex_subscripts.py docs/source/1_euclidean_gas/your_document.md
```

---

## Build and Preview

### Build documentation
```bash
# Build Jupyter Book
make build-docs

# Serve locally
make serve-docs

# Open browser to http://localhost:8000
```

### Check for errors
```bash
# Build and look for warnings
make build-docs 2>&1 | grep -E "WARNING|ERROR"
```

---

## Dual-Review Workflow

### Submit to both reviewers (in Claude Code)

**Gemini 2.5 Pro:**
```
Use mcp__gemini-cli__ask-gemini with model: "gemini-2.5-pro"

Prompt: [paste review request]
```

**Codex:**
```
Use mcp__codex__codex

Prompt: [paste IDENTICAL review request]
```

**Compare results:**
1. Consensus issues (both agree) → High confidence, implement
2. Contradictions (disagree) → Verify manually against framework docs
3. Unique issues (only one) → Verify before accepting

---

## Common Fixes

### Math not rendering
```bash
# Missing blank line before $$
python src/tools/fix_math_formatting.py your_document.md
```

### Broken cross-reference
```bash
# Find correct label
cat docs/glossary.md | grep "your-concept"

# Use correct syntax
{prf:ref}`thm-correct-label`
```

### Build errors
```bash
# Check directive syntax (should be ::: not ::::)
# Check labels exist
# Check math formatting

# Rebuild
make build-docs
```

---

## Notation Quick Reference

### Greek Letters
```latex
$\gamma$  (friction)
$\beta$   (exploitation weight)
$\sigma$  (perturbation noise)
$\tau$    (time step)
$\epsilon_F$ (regularization)
```

### Calligraphic Letters
```latex
$\mathcal{X}$  (state space)
$\mathcal{Y}$  (algorithmic space)
$\mathcal{S}$  (swarm configuration)
$\mathcal{A}$  (alive set)
```

### Common Operators
```latex
$d_\mathcal{X}$  (metric on state space)
$\phi$           (projection map)
$\psi$           (squashing map)
$\Psi_{\text{clone}}$  (cloning operator)
$\Psi_{\text{kin}}$    (kinetic operator)
```

---

## Quality Checklist (Before Commit)

```
[ ] Consulted docs/glossary.md for existing definitions
[ ] Used proper MyST directive syntax (:::)
[ ] All math uses $ / $$ (not backticks)
[ ] Blank line before all $$ blocks
[ ] All directives have :label: fields
[ ] Labels use lowercase kebab-case
[ ] Cross-references use {prf:ref}
[ ] Notation matches glossary
[ ] Proofs are complete
[ ] Submitted to dual-review (Gemini + Codex)
[ ] Addressed CRITICAL/MAJOR feedback
[ ] Verified feedback against framework docs
[ ] Ran formatting tools
[ ] Build succeeds with no warnings
[ ] Previewed rendering locally
```

---

## Complete Example Workflow

### Write a new theorem about cloning convergence

```bash
# 1. Check if related content exists
cat docs/glossary.md | grep -i "cloning" | grep "thm-"

# 2. Draft theorem in document
vim docs/source/1_euclidean_gas/03_cloning.md

# Add:
# :::{prf:theorem} Cloning Convergence Rate
# :label: thm-cloning-convergence-rate
# ...
# :::

# 3. Format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/03_cloning.md

# 4. Dual-review (in Claude Code)
# Submit to mcp__gemini-cli__ask-gemini (model: gemini-2.5-pro)
# Submit to mcp__codex__codex
# Use IDENTICAL prompt for both

# 5. Implement feedback
vim docs/source/1_euclidean_gas/03_cloning.md

# 6. Re-format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/03_cloning.md

# 7. Build and verify
make build-docs
make serve-docs

# 8. Check rendering at http://localhost:8000
```

---

**Time estimates:**
- Simple definition: ~15 min (draft + format + review + build)
- Theorem with proof: ~1-2 hours (draft + dual-review + fixes + build)
- Formatting fixes: ~2 min (run tools + rebuild)

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [WORKFLOW.md](./WORKFLOW.md) - Step-by-step procedures
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
