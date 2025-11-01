# Mathematical Writing - Complete Workflow

Detailed step-by-step procedures for writing and publishing mathematical documentation.

---

## Prerequisites

- ✅ Python environment with fragile package installed
- ✅ Access to Gemini 2.5 Pro API (for dual-review)
- ✅ Access to Codex (for dual-review)
- ✅ Jupyter Book installed (`make install`)
- ✅ Familiarity with MyST markdown syntax

---

## Complete Writing Pipeline

### Stage 0: Research and Planning

**Purpose**: Understand existing framework before writing

#### Step 0.1: Consult Mathematical Index

```bash
# Always start here
cat docs/glossary.md | head -100

# Search for related content
cat docs/glossary.md | grep -i "your-topic"

# Search by tags
cat docs/glossary.md | grep "tag:cloning"
cat docs/glossary.md | grep "tag:langevin"
cat docs/glossary.md | grep "tag:convergence"

# Find theorems
cat docs/glossary.md | grep "thm-" | grep "convergence"

# Find definitions
cat docs/glossary.md | grep "obj-" | grep "gas"
```

**What to look for:**
- Existing definitions of your concept
- Related theorems you need to reference
- Notation conventions to follow
- Source documents to read

#### Step 0.2: Read Source Documents

```bash
# Find the full statement in source document
# (glossary provides document references)

# Example: Reading about cloning
cat docs/source/1_euclidean_gas/03_cloning.md | less

# For large files, use offset/limit
# Read lines 1-100
head -100 docs/source/1_euclidean_gas/03_cloning.md

# Read specific section (lines 200-300)
sed -n '200,300p' docs/source/1_euclidean_gas/03_cloning.md
```

#### Step 0.3: Identify Dependencies

From glossary and source documents, identify:
- [ ] Mathematical objects you'll use
- [ ] Axioms you'll invoke
- [ ] Theorems you'll reference
- [ ] Notation you'll need

---

### Stage 1: Drafting Content

**Purpose**: Create initial mathematical content with proper structure

#### Step 1.1: Choose Document Location

```bash
# Determine which document your content belongs in

# Core framework axioms
docs/source/1_euclidean_gas/01_fragile_gas_framework.md

# Euclidean Gas specification
docs/source/1_euclidean_gas/02_euclidean_gas.md

# Cloning operator details
docs/source/1_euclidean_gas/03_cloning.md

# Convergence mathster
docs/source/1_euclidean_gas/04_convergence.md

# Create new document if needed
vim docs/source/1_euclidean_gas/10_your_new_topic.md
```

#### Step 1.2: Draft Definition

**Template:**
```markdown
:::{prf:definition} Your Concept Name
:label: obj-your-concept

**Informal Description**:

[Provide intuitive explanation in plain language]

**Formal Definition**:

Let $\mathcal{X}$ be the state space. We define...

$$
\text{mathematical expression}
$$

where $x \in \mathcal{X}$ is...

**Notation**:
- $\mathcal{X}$: State space
- $x$: Position vector
- $d_\mathcal{X}$: Metric on $\mathcal{X}$

**Properties** (if applicable):
1. [Property 1 with brief justification]
2. [Property 2 with brief justification]
:::

:::{note}
**Physical Intuition**: [Explain the physical meaning or interpretation]
:::
```

**Key requirements:**
- Use `obj-` prefix for definition labels
- Include both informal and formal descriptions
- Define all notation explicitly
- Add physical intuition in admonitions

#### Step 1.3: Draft Theorem

**Template:**
```markdown
:::{prf:theorem} Your Theorem Name
:label: thm-your-theorem

**Statement**:

Under the following hypotheses, we establish...

**Hypotheses**:
1. $(\mathcal{X}, d_\mathcal{X})$ is a complete metric space
2. The cloning operator satisfies {prf:ref}`axiom-bounded-cloning`
3. [Additional assumptions]

**Conclusion**:

There exists a constant $C > 0$ such that...

$$
\text{main result}
$$
:::

:::{important}
This theorem requires {prf:ref}`axiom-name` and {prf:ref}`lem-helper`.
:::
```

#### Step 1.4: Draft Proof

**Template:**
```markdown
:::{prf:proof}
We proceed in [number] steps.

**Step 1: [Establish foundation]**

By {prf:ref}`def-state-space`, we have...

$$
x \in \mathcal{X} \implies d_\mathcal{X}(x, y) < \infty
$$

This follows from [justification].

**Step 2: [Main derivation]**

Consider the operator $\Psi_{\text{clone}}$. From {prf:ref}`axiom-bounded-cloning`, we obtain

$$
\|\Psi_{\text{clone}}(x) - \Psi_{\text{clone}}(y)\| \leq L \cdot d_\mathcal{X}(x, y)
$$

where $L$ is the Lipschitz constant.

**Step 3: [Intermediate result]**

Applying {prf:ref}`lem-helper-lemma` with $\epsilon = 1/N$, we get

$$
\mathbb{E}[\text{error}] \leq C \cdot N^{-1/2}
$$

**Step 4: [Conclude]**

Combining Steps 2 and 3 yields the desired bound:

$$
\text{final result}
$$

This completes the proof. ∎
:::
```

**Key requirements:**
- Break proof into logical steps
- Justify every claim with references
- Show all mathematical derivations
- Use cross-references (`{prf:ref}`)

#### Step 1.5: Add Supporting Elements

**Admonitions for physical intuition:**
```markdown
:::{note}
**Physical Interpretation**:
This bound means that walkers cannot drift arbitrarily far from the swarm.
:::

:::{tip}
**Computational Note**:
In practice, choose $N \geq 100$ for the bound to be effective.
:::

:::{important}
**Framework Requirement**:
This lemma relies critically on {prf:ref}`axiom-geometric-consistency`.
:::

:::{warning}
**Common Pitfall**:
Do not confuse $d_\mathcal{X}$ (state space metric) with $d_{\text{alg}}$ (algorithmic distance).
:::
```

---

### Stage 2: Formatting and Validation

**Purpose**: Ensure LaTeX/MyST compliance and notation consistency

#### Step 2.1: Save Draft

```bash
# Save your work
vim docs/source/1_euclidean_gas/your_document.md
# Write content, save (:wq)
```

#### Step 2.2: Run Formatting Tools

**All-in-one fix:**
```bash
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md
```

**Or run individually:**
```bash
# 1. Convert Unicode to LaTeX
python src/tools/convert_unicode_math.py docs/source/1_euclidean_gas/your_document.md

# 2. Convert backticks to dollar signs
python src/tools/convert_backticks_to_math.py docs/source/1_euclidean_gas/your_document.md

# 3. Fix spacing before $$ blocks (CRITICAL)
python src/tools/fix_math_formatting.py docs/source/1_euclidean_gas/your_document.md

# 4. Handle complex subscripts
python src/tools/fix_complex_subscripts.py docs/source/1_euclidean_gas/your_document.md
```

**What each tool fixes:**
- **convert_unicode_math**: α → \alpha, ∇ → \nabla, ∈ → \in
- **convert_backticks_to_math**: `x` → $x$, `gamma` → $\gamma$
- **fix_math_formatting**: Adds blank line before `$$` blocks
- **fix_complex_subscripts**: `x_i_j` → `x_{i,j}`

#### Step 2.3: Verify Directive Syntax

```bash
# Check for common errors
grep -n "::::" docs/source/1_euclidean_gas/your_document.md  # Should use ::: not ::::

# Check for missing labels
grep -A 5 "{prf:definition}" docs/source/1_euclidean_gas/your_document.md | grep ":label:"

# Check label format (should be lowercase kebab-case)
grep ":label:" docs/source/1_euclidean_gas/your_document.md | grep "[A-Z_]"  # Should be empty
```

#### Step 2.4: Test Build

```bash
# Build documentation
make build-docs

# Look for warnings/errors
make build-docs 2>&1 | grep -E "WARNING|ERROR"
```

**Common build errors:**
- Missing blank line before `$$`
- Invalid directive syntax (:::: instead of :::)
- Broken cross-references
- Missing labels

**Fix and rebuild:**
```bash
# Fix issues
vim docs/source/1_euclidean_gas/your_document.md

# Re-run formatting
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md

# Rebuild
make build-docs
```

---

### Stage 3: Dual-Review Validation

**Purpose**: Critical review for rigor, consistency, and correctness

#### Step 3.1: Prepare Review Sections

**For theorems/proofs:**
```bash
# Extract theorem + proof for review
sed -n '/prf:theorem.*thm-your-theorem/,/:::{prf:proof/p' docs/source/1_euclidean_gas/your_document.md > /tmp/theorem_section.md
sed -n '/prf:proof/,/:::/p' docs/source/1_euclidean_gas/your_document.md >> /tmp/theorem_section.md

# Copy to clipboard or read
cat /tmp/theorem_section.md
```

**For definitions:**
```bash
# Extract definition for review
sed -n '/prf:definition.*obj-your-concept/,/:::/p' docs/source/1_euclidean_gas/your_document.md > /tmp/definition_section.md

cat /tmp/definition_section.md
```

#### Step 3.2: Draft Review Prompt

**Standard review prompt:**
```
Review this mathematical content for:

1. **Rigor**: Are all claims proven? Are proofs complete?
2. **Consistency**: Does notation match docs/glossary.md?
3. **Correctness**: Are derivations mathematically sound?
4. **Clarity**: Is the presentation well-organized?

Content to review:

[Paste theorem/definition/proof here]

Focus areas:
- [Specific area 1]
- [Specific area 2]
- [Specific area 3]

Please provide:
- Critical issues (severity ratings)
- Specific line references
- Proposed fixes with justification
```

#### Step 3.3: Submit to Both Reviewers

**In Claude Code, use IDENTICAL prompts:**

**Reviewer 1: Gemini 2.5 Pro**
```
Tool: mcp__gemini-cli__ask-gemini
Parameters:
  model: "gemini-2.5-pro"
  prompt: [paste review prompt]
```

**Reviewer 2: Codex**
```
Tool: mcp__codex__codex
Parameters:
  prompt: [paste IDENTICAL review prompt]
```

**CRITICAL**: Run both in parallel (single message with two tool calls) and use identical prompts.

#### Step 3.4: Compare Review Results

**Wait for both reviews to complete**, then analyze:

**1. Identify Consensus Issues** (both reviewers agree):
```
Example:
- Gemini: "Step 3 lacks justification for bound"
- Codex: "Step 3 derivation has gap - bound not proven"

→ HIGH CONFIDENCE: This is a real issue, prioritize fixing
```

**2. Identify Contradictions** (reviewers disagree):
```
Example:
- Gemini: "Notation $d_\mathcal{X}$ inconsistent with framework"
- Codex: "Notation $d_\mathcal{X}$ matches standard framework usage"

→ VERIFY MANUALLY: Check docs/glossary.md to determine correct usage
```

**3. Identify Unique Issues** (only one reviewer identifies):
```
Example:
- Gemini: [no issue]
- Codex: "Consider adding reference to Lemma 4.2 for clarity"

→ MEDIUM CONFIDENCE: Verify against framework docs before accepting
```

#### Step 3.5: Verify Feedback Against Framework

**For each issue, verify against source documents:**

```bash
# Check notation consistency
cat docs/glossary.md | grep "d_\mathcal{X}"

# Check if suggested axiom exists
cat docs/glossary.md | grep "axiom-suggested-name"

# Read full context in source document
cat docs/source/1_euclidean_gas/01_fragile_gas_framework.md | grep -A 20 "axiom-suggested-name"
```

**Critical evaluation questions:**
- [ ] Does the issue reference real framework entities?
- [ ] Is the suggested fix mathematically correct?
- [ ] Does the fix preserve algorithmic intent?
- [ ] Are there alternative solutions?

#### Step 3.6: Implement Fixes

**Priority order:**
1. **CRITICAL issues** (both reviewers agree + verified against framework)
2. **MAJOR issues** (consensus or strong evidence)
3. **MINOR issues** (clarity improvements)

**For each fix:**
```bash
# 1. Edit document
vim docs/source/1_euclidean_gas/your_document.md

# 2. Re-format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md

# 3. Rebuild
make build-docs

# 4. Verify fix
cat docs/source/1_euclidean_gas/your_document.md | grep -A 10 "fixed-section"
```

#### Step 3.7: Re-Review (if needed)

**If CRITICAL issues were addressed:**

```
Submit to both reviewers again:

"Previous review identified issue X. I've implemented the following fix:

[Describe fix]

Please verify:
1. Does this fix resolve the issue?
2. Are there any new issues introduced?
3. Is the mathematical reasoning sound?

Updated content:
[Paste updated section]
"
```

---

### Stage 4: Building and Publishing

**Purpose**: Generate publication-ready documentation

#### Step 4.1: Final Formatting Pass

```bash
# Comprehensive formatting
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md

# Convert mermaid blocks (if present)
python src/tools/convert_mermaid_blocks.py docs/source/1_euclidean_gas/your_document.md --in-place
```

#### Step 4.2: Clean Build

```bash
# Clean previous build
make clean

# Fresh build
make build-docs

# Check for warnings
make build-docs 2>&1 | tee build_log.txt
grep -E "WARNING|ERROR" build_log.txt
```

#### Step 4.3: Local Preview

```bash
# Serve documentation
make serve-docs

# Open browser to http://localhost:8000
# Navigate to your document
```

**What to check:**
- [ ] Math renders correctly (no raw LaTeX)
- [ ] Cross-references are clickable blue links
- [ ] Theorems/definitions have proper numbering
- [ ] Admonitions render with correct styling
- [ ] Equations are centered and well-spaced

#### Step 4.4: Final Quality Check

**Manual verification:**
```bash
# Check math rendering
# - Open http://localhost:8000/path/to/your_document.html
# - Verify all $...$ and $$...$$ render correctly

# Check cross-references
# - Click all {prf:ref} links
# - Verify they navigate to correct definitions/theorems

# Check theorem numbering
# - Verify theorems are numbered correctly
# - Check that references show correct numbers
```

**Checklist:**
- [ ] All math renders without errors
- [ ] All cross-references work (blue links)
- [ ] Theorem/definition numbers are correct
- [ ] Proofs are complete and readable
- [ ] Admonitions render correctly
- [ ] No build warnings/errors

---

## Iteration and Revision

### Handling Feedback

**If reviewer suggests major restructuring:**

1. **Understand the issue fully**:
   - Read both Gemini and Codex feedback
   - Check framework documents
   - Identify root cause

2. **Propose alternative if you disagree**:
   - Document your reasoning
   - Provide mathematical justification
   - Reference framework axioms
   - Present to user for decision

3. **Implement accepted changes**:
   - Edit document
   - Re-format
   - Re-build
   - Re-review (if critical)

### Common Revision Scenarios

**Scenario 1: Add missing lemma**

```bash
# Draft lemma
vim docs/source/1_euclidean_gas/your_document.md

# Add before theorem that uses it:
# :::{prf:lemma} Helper Result
# :label: lem-helper-result
# ...
# :::

# Format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md

# Rebuild
make build-docs
```

**Scenario 2: Fix notation inconsistency**

```bash
# Find all uses of inconsistent notation
grep -n "old_notation" docs/source/1_euclidean_gas/your_document.md

# Replace (careful with regex)
sed -i 's/old_notation/new_notation/g' docs/source/1_euclidean_gas/your_document.md

# Re-format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md

# Rebuild
make build-docs
```

**Scenario 3: Expand proof step**

```bash
# Edit document
vim docs/source/1_euclidean_gas/your_document.md

# Add detailed derivation:
# **Step 2a: [Intermediate result]**
#
# By [justification], we have
#
# $$
# \text{intermediate equation}
# $$
#
# **Step 2b: [Continue derivation]**
#
# Applying [technique] yields...

# Format
python src/tools/format_math_blocks.py docs/source/1_euclidean_gas/your_document.md

# Rebuild
make build-docs
```

---

## Time Estimates

| Task | Time | Notes |
|------|------|-------|
| Research (glossary + docs) | ~15-30 min | Essential first step |
| Draft definition | ~30-60 min | With proofs of properties |
| Draft theorem | ~1-2 hours | Without proof |
| Draft proof | ~2-4 hours | Complex proofs take longer |
| Formatting | ~5 min | Automated tools |
| Test build | ~2 min | Quick verification |
| Dual-review | ~20-40 min | Parallel execution |
| Implement feedback | ~30-60 min | Depends on severity |
| Final build + preview | ~5 min | Quality check |

**Total for theorem + proof**: ~4-8 hours

---

**Related:**
- [SKILL.md](./SKILL.md) - Complete documentation
- [QUICKSTART.md](./QUICKSTART.md) - Copy-paste commands
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues
